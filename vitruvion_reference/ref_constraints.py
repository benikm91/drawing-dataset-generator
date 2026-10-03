"""Vitruvion's own constraint tokens for every sketch of a sg_filtered_unique.npy, written next to it."""
import sys
import numpy as np
from sketchgraphs import data as datalib
from sketchgraphs.data import flat_array
from img2cad import data_utils, dataset
# Only a type annotation in constraint_data; SketchGraphs master no longer defines it (Vitruvion issue #7).
datalib.ConstructionSequence = list
from img2cad.constraint_data import tokenize_constraints

path = sys.argv[1]
seqs = flat_array.load_dictionary_flat(path)['sequences']
out = []
for i in range(len(seqs)):
    seq = seqs[i]
    sketch = datalib.sketch_from_sequence(seq)
    data_utils.normalize_sketch(sketch)
    _, gather = dataset.tokenize_sketch(sketch, 64)
    out.append(tokenize_constraints(seq, gather)['val'])
offsets = np.concatenate([[0], np.cumsum([len(v) for v in out])])
np.savez(path.replace('.npy', '_constraints.npz'), val=np.concatenate(out), offsets=offsets)
print('wrote constraint tokens for', len(out), 'sketches')
