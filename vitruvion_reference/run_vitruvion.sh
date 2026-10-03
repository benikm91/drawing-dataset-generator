#!/usr/bin/env bash
# Builds the Vitruvion SketchGraphs dataset with create_vitruvion_dataset.py, and checks it against
# Vitruvion's own pipeline run unchanged on the same shards.
#
#   vitruvion_reference/run_vitruvion.sh all          # everything below, in order
#   vitruvion_reference/run_vitruvion.sh download     # the 128 raw JSON shards (43 GB), resumable
#   vitruvion_reference/run_vitruvion.sh build        # the reference container
#   vitruvion_reference/run_vitruvion.sh ours         # our dataset, into $OUTPUT_DIR
#   vitruvion_reference/run_vitruvion.sh reference    # Vitruvion's pipeline, into $REFERENCE_DIR
#   vitruvion_reference/run_vitruvion.sh compare      # the two, sketch by sketch
#   vitruvion_reference/run_vitruvion.sh render       # Vitruvion's renders of $RELEASE_DIR, resumable
#
# `all` runs ours and the reference side by side, then compares them. Logs go to $LOG_DIR.
# Set SHARDS=1 (or "1 2 3") to run on those shards only, as a rehearsal; RENDER_ONLY="1 2" renders
# only those of the $RENDER_CHUNKS chunks.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="${CACHE:-$HOME/.detr-cache}"
SHARDS_DIR="${SHARDS_DIR:-$CACHE/input/sketchgraphs/shards}"
WORK_DIR="${WORK_DIR:-$CACHE/vitruvion-ref}"
SHARDS="${SHARDS:-$(seq 1 128)}"
SUFFIX="${SUFFIX:-full}"
if [ "$SUFFIX" = full ]; then DEFAULT_OUTPUT="$REPO/dataset_vitruvion"; else DEFAULT_OUTPUT="$REPO/dataset_vitruvion_$SUFFIX"; fi
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT}"
REFERENCE_DIR="${REFERENCE_DIR:-$WORK_DIR/out/$SUFFIX}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/logs}"
RELEASE_DIR="${RELEASE_DIR:-$REPO/dataset_sketches_vitruvion}"
RENDER_CHUNKS="${RENDER_CHUNKS:-16}"
NUM_NOISY="${NUM_NOISY:-5}"
IMAGE=vitruvion-ref

VITRUVION_COMMIT=1b91fff5597b3a4e272e6490e4d458f3ef790e62
SKETCHGRAPHS_COMMIT=1f27f5f9459926d38318007c71b72697083b2f3c
SOURCE=https://sketchgraphs.cs.princeton.edu/shards

mkdir -p "$LOG_DIR"
shard_name() { printf "shard_%03d_of_128.tar.zst" "$1"; }

download() {
    mkdir -p "$SHARDS_DIR"
    for shard in $SHARDS; do
        name=$(shard_name "$shard")
        [ -f "$SHARDS_DIR/$name" ] && continue
        echo "$name"
    done | xargs -P 4 -I{} sh -c \
        'curl -sS --fail -C - -o "$1/{}.part" "$2/{}" && mv "$1/{}.part" "$1/{}" && echo "downloaded {}"' \
        _ "$SHARDS_DIR" "$SOURCE"
    for shard in $SHARDS; do
        [ -f "$SHARDS_DIR/$(shard_name "$shard")" ] || { echo "missing $(shard_name "$shard")"; exit 1; }
    done
    echo "shards present: $(echo $SHARDS | wc -w | tr -d ' ') in $SHARDS_DIR ($(du -sh "$SHARDS_DIR" | cut -f1))"
}

build() {
    command -v docker >/dev/null || { echo "docker is not installed"; exit 1; }
    docker info >/dev/null 2>&1 || { echo "docker is not running: start Docker Desktop"; exit 1; }
    mkdir -p "$WORK_DIR/build"
    for spec in "vitruvion PrincetonLIPS/vitruvion $VITRUVION_COMMIT" \
                "SketchGraphs PrincetonLIPS/SketchGraphs $SKETCHGRAPHS_COMMIT"; do
        set -- $spec
        [ -d "$WORK_DIR/build/$1" ] || git clone -q "https://github.com/$2.git" "$WORK_DIR/build/$1"
        git -C "$WORK_DIR/build/$1" checkout -q "$3"
    done
    cp "$REPO/vitruvion_reference/Dockerfile" "$WORK_DIR/build/Dockerfile"
    docker build -q -t "$IMAGE" "$WORK_DIR/build"
}

ours() {
    cd "$REPO"
    # shellcheck disable=SC2086
    uv run create_vitruvion_dataset.py --output-dir "$OUTPUT_DIR" --shards-dir "$SHARDS_DIR" \
        --shards $SHARDS --overwrite
}

reference() {
    # Vitruvion reads every file in its input folder as a shard, so it is given a folder holding
    # exactly the shards asked for, linked rather than copied.
    input="$WORK_DIR/input/$SUFFIX"
    rm -rf "$input" && mkdir -p "$input" "$REFERENCE_DIR"
    for shard in $SHARDS; do ln "$SHARDS_DIR/$(shard_name "$shard")" "$input/"; done
    run() { docker run --rm -v "$input":/shards:ro -v "$REFERENCE_DIR":/out \
                -v "$REPO/vitruvion_reference":/tools:ro "$IMAGE" "$@"; }
    run python -m img2cad.pipeline.filter_sequences_from_source \
        input_folder=/shards output_folder=/out total_sketches=16261381 hydra.run.dir=/out/hydra_filter
    run python -m img2cad.pipeline.tokenize_sequences \
        sequence_file=/out/sg_filtered.npy hydra.run.dir=/out/hydra_tokenize
    run python /tools/ref_constraints.py /out/sg_filtered_unique.npy
}

render() {
    # Vitruvion's renderer on the released sequences, as their cluster ran it
    # (img2cad/pipeline/render_images.sh): one clean and $NUM_NOISY hand-drawn renders per sketch,
    # 128 px, split into chunks named like their published render_p128_XX_of_16.npy. A chunk
    # already written is skipped, so the step can be stopped and restarted.
    mkdir -p "$RELEASE_DIR/renders"
    for chunk in ${RENDER_ONLY:-$(seq 1 "$RENDER_CHUNKS")}; do
        name=$(printf "render_p128_%02d_of_%02d.npy" "$chunk" "$RENDER_CHUNKS")
        partial="${name%.npy}.part.npy"   # np.save appends .npy to any other name
        [ -f "$RELEASE_DIR/renders/$name" ] && { echo "have $name"; continue; }
        echo "rendering $name"
        docker run --rm -v "$RELEASE_DIR":/data -v "$REPO/vitruvion_reference":/tools:ro "$IMAGE" sh -c "
            python /tools/patch_prerender.py &&
            python -m img2cad.pipeline.prerender_images sequence_file=/data/sg_filtered_unique.npy \
                num_noisy_samples=$NUM_NOISY slurm_array_task_id=$((chunk - 1)) \
                slurm_array_task_count=$RENDER_CHUNKS output_file=/data/renders/$partial \
                hydra.run.dir=/tmp/hydra" 2>&1 | tr '\r' '\n' | grep -v 'it/s\]$' | grep -v Warning
        mv "$RELEASE_DIR/renders/$partial" "$RELEASE_DIR/renders/$name"
    done
}

compare() {
    cd "$REPO"
    uv run python vitruvion_reference/compare.py "$REFERENCE_DIR" "$OUTPUT_DIR"
}

case "${1:-}" in
    download|build|ours|reference|compare|render)
        "$1" 2>&1 | tee "$LOG_DIR/$1-$SUFFIX.log" ;;
    all)
        download 2>&1 | tee "$LOG_DIR/download-$SUFFIX.log"
        build 2>&1 | tee "$LOG_DIR/build-$SUFFIX.log"
        echo "running ours and the reference side by side; logs in $LOG_DIR"
        ours > "$LOG_DIR/ours-$SUFFIX.log" 2>&1 & ours_pid=$!
        reference > "$LOG_DIR/reference-$SUFFIX.log" 2>&1 & reference_pid=$!
        wait $ours_pid || { echo "ours failed: see $LOG_DIR/ours-$SUFFIX.log"; exit 1; }
        echo "ours done: $(grep -c . "$OUTPUT_DIR/sketches.jsonl") sketches"
        wait $reference_pid || { echo "reference failed: see $LOG_DIR/reference-$SUFFIX.log"; exit 1; }
        echo "reference done"
        compare 2>&1 | tee "$LOG_DIR/compare-$SUFFIX.log" ;;
    *)
        sed -n 2,15p "$0"; exit 2 ;;
esac
