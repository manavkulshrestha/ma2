from datetime import datetime
import pickle


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
    return zip(*[lst[i:] for i in range(n)])

def dict_np_equal(d1, d2):
    return all([(k1 == k2 and (v1 == v2).all()) for ((k1,v1), (k2,v2)) in zip(d1.items(), d2.items())])