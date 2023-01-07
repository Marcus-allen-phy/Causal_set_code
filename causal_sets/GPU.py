import cupy as cp

def interval_size_gpu(rels_above,rels_below):
    rels_above_gpu = cp.asarray(rels_above)
    rels_below_gpu = cp.asarray(rels_below)
    return cp.asnumpy(cp.bincount(cp.matmul(rels_below_gpu,rels_above_gpu, dtype = cp.ushort).astype(cp.int64).flatten())[2:])