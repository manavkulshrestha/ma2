from datetime import datetime
import pickle
from itertools import islice
import numpy as np

def save_pkl(obj, s, ext=False):
    with open(f'{s}.pkl' if ext else s, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(s, ext=False):
    with open(f'{s}.pkl' if ext else s, 'rb') as f:
        return pickle.load(f)
    
def time_label():
    return datetime.now().strftime('%y-%m-%d-%H%M%S%f')[:17]

def sliding(lst, n):
    """ returns a sliding window of size n over a list lst """
    return zip(*[islice(lst, i) for i in range(n)])

def chunked(lst, n, exclude_less=True):
    """ return chunks of the list of size n. last chunk discarded if < n """
    chunks = [lst[i:i + n] for i in range(0, len(lst), n)]
    if exclude_less and len(chunks[-1]) < n:
        chunks.pop()

    return chunks

def dict_np_equal(d1, d2):
    return all([(k1 == k2 and (v1 == v2).all()) for ((k1,v1), (k2,v2)) in zip(d1.items(), d2.items())])

def pdisp(X):
    return X[:, np.newaxis, :] - X[np.newaxis, :, :]