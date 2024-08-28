import dill
import numpy as np
from copy import copy

with open('./data/full_classReady.dill', 'rb') as f:
    data = dill.load(f)

x = copy(data['x'])
y = data['y']
subj = data['subj']
us = np.unique(subj)
m, std = [], []
for i, s in enumerate(us):
    idx = np.where(subj == s)[0]
    xs = x[idx]
    xs[xs==0] = np.nan
    mS = np.nanmean(xs, axis=(0, 3), keepdims=True)
    stdS = np.nanstd(xs, axis=(0, 3), keepdims=True)
    x[idx] = (xs - mS) / stdS
    m.append(np.squeeze(mS))
    std.append(np.squeeze(stdS))


m = np.stack(m)
std = np.stack(std)
x[np.isnan(x)] = 0
data['x'] = x
with open('./data/full_classReady_norm.dill', 'wb') as f:
    data = dill.dump(data, f)
