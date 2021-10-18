import numpy as np
from scipy import sparse
import scipy
from scipy.sparse import random
from scipy import stats

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return  loader['data'], loader['indices'], loader['indptr'],loader['shape']

class CustomRandomState(np.random.RandomState):
    def randint(self, k):
        i = np.random.randint(k)
        return i - i % 2

'''Please decide your meta data below'''

a_deg = 64 #number of products
feat_deg = 32 #degree of node feature
hidden_dim = 32 #dimension of hidden layer 1
class_dim = 16 #number of categories

np.random.seed(12345)
rs = CustomRandomState()
rvs = stats.poisson(25, loc=10).rvs
S = random(a_deg, a_deg, density=0.15, random_state=rs, data_rvs=rvs)
adj_matrix = S.A
adj_matrix/=100
adj_matrix = adj_matrix.astype(np.float32)

dense_64 = np.random.rand(a_deg,feat_deg)
dense_64 = dense_64*2
dense = dense_64.astype(np.float32)

AX = np.matmul(adj_matrix, dense)
AX = AX.astype(np.float32)

size = (feat_deg, hidden_dim)
w1 = np.random.uniform(-1, 1, size)
w1 = w1.astype(np.float32)

size2 = (hidden_dim, class_dim)
w2 = np.random.uniform(-2, 2, size2)
w2 = w2.astype(np.float32)

H = np.matmul(AX,w1)
H = H.astype(np.float32)
H = np.maximum(H, 0)

AH = np.matmul(adj_matrix,H)
AH = AH.astype(np.float32)

final = np.matmul(AH,w2)
final = final.astype(np.float32)

b=sparse.csr_matrix(adj_matrix)
save_sparse_csr('sparse', b)
data_64, indicies, indptr, shape_64 = load_sparse_csr('sparse.npz')
print('NNZ:', len(data_64))
data = data_64.astype(np.float32)
shape_64 = np.array([shape_64[0], len(data)])
shape = shape_64.astype(np.int32)

data.tofile('data.bin')
AX.tofile('AX.bin')
indptr.tofile('indptr.bin')
indicies.tofile('indices.bin')
shape.tofile('shape.bin')
dense.tofile('feats.bin')
w1.tofile('w1.bin')
w2.tofile('w2.bin')
H.tofile('H.bin')