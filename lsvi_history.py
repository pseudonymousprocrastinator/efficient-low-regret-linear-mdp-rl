import numpy as np

class LSVIHistory:
    def __init__(self, int_size, threshold, H, d):
        self.int_size = int_size
        self.H = H
        self.d = d
        if threshold is None:
            self.threshold = 0.1
        else:
            self.threshold = threshold

        self.buffer = np.zeros(shape=(H, int_size, d*d))
        self.is_close = np.full(shape=(H, int_size, int_size), fill_value=False, dtype=bool)
        self.idxs = np.zeros(shape=(H,), dtype=np.int32)
        self.sizes = np.zeros(shape=(H,), dtype=np.int32)
    # End fn __init__
    
    def clear(self):
        self.buffer.fill(0.)
        self.is_close.fill(False)
        self.idxs.fill(0)
        self.sizes.fill(0)
    # End fn clear
    
    def add(self, h, M):
        self.buffer[h, self.idxs[h]] = M.flatten()
        for i in range(self.int_size):
            isc = bool(np.linalg.norm(self.buffer[h, i] - self.buffer[h, self.idxs[h]]) <= self.threshold*self.d*self.d)
            self.is_close[h, self.idxs[h], i] = self.is_close[h, i, self.idxs[h]] = isc
        self.idxs[h] += 1
        self.sizes[h] = min(self.int_size, self.sizes[h] + 1)
        if self.idxs[h] >= self.int_size:
            self.idxs[h] = 0
    # End fn add
    
    def size(self, h):
        return self.sizes[h]
    # End fn size
     
    def learning_cond(self, h):
        return not(np.all(self.is_close))
    # End fn learning_cond
    
    def __sizeof__(self):
        return 16 + self.buffer.nbytes + self.is_close.nbytes + 2*self.idxs.nbytes
    # End fn __sizeof__
# End class LSVIHistory
