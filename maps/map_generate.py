import numpy as np

def map_generate(size,num_ob):
    map = np.zeros(size)
    seed = np.random.choice([0,1],1)
    row_ind = np.random.choice(size[0],num_ob,replace=seed)
    col_ind = np.random.choice(size[1],num_ob,replace=seed)
    map[row_ind,col_ind] = 1
    return map

if __name__ == '__main__':
    map = map_generate([4,4],3)
    print(map)