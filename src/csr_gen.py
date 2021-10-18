import numpy as np
from scipy import sparse
import scipy

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return  loader['data'], loader['indices'], loader['indptr'],loader['shape']

a=np.array([[1,0,0,0,2,1,0,0,0,1],
            [9,0,1,0,0,0,3,0,0,0],
            [0,0,0,0,0,0,0,4,0,0],
            [0,0,5,0,0,0,0,0,0,0],
            ])

b=sparse.csr_matrix(a)
save_sparse_csr('sparse', b)
data, indicies, indptr, shape = load_sparse_csr('sparse.npz')

print(data)
print(indicies)
print(indptr)
print(shape)

data.tofile('data.bin')
indptr.tofile('indptr.bin')
a.tofile('a.bin')