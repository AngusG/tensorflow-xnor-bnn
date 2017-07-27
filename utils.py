import os.path
import numpy as np

def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        dir += '/1'
        os.makedirs(dir)
    else:
        sub_dirs = next(os.walk(dir))[1]
        if len(sub_dirs) > 0:
            print(dir)
            arr = np.asarray(sub_dirs).astype('int')
            sub = str(arr.max() + 1)
            print(sub)
            dir += '/' + sub
            print(dir)
        else:
            dir += '/1'
        os.makedirs(dir)
    print('Logging to %s' % dir)
    return dir