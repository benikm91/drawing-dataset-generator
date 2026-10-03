"""The split Vitruvion's own code draws from a sg_filtered_unique.npy, as index lists.

Vitruvion splits at training time (`img2cad.primitives_data.split_dataset`: torch.randperm with
seed 4242424242, 92.5% train, 2.5% val, 5% test). This runs that function, in the reference
container's torch 1.9, and writes the indices it draws, so the split can be used without it.

    python ref_split.py /out/sg_filtered_unique.npy /out/splits.json
"""
import json
import sys

from sketchgraphs import data as datalib
from sketchgraphs.data import flat_array

# Only a type annotation in img2cad; SketchGraphs master no longer defines it (Vitruvion issue #7).
datalib.ConstructionSequence = list
from img2cad.primitives_data import split_dataset, PrimitiveDataConfig

count = len(flat_array.load_dictionary_flat(sys.argv[1])["sequences"])
config = PrimitiveDataConfig()
train, val, test = split_dataset(list(range(count)), config.validation_fraction, config.test_fraction)
splits = {
    "procedure": "img2cad.primitives_data.split_dataset (torch.randperm, seed 4242424242)",
    "fractions": {"val": config.validation_fraction, "test": config.test_fraction},
    "train": sorted(train.indices.tolist()),
    "val": sorted(val.indices.tolist()),
    "test": sorted(test.indices.tolist()),
}
with open(sys.argv[2], "w") as handle:
    json.dump(splits, handle)
print({name: len(splits[name]) for name in ("train", "val", "test")}, "of", count)
