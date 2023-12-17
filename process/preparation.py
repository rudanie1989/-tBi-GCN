
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:42:00 2021

@author: S3575040

#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csgraph.html 
"""
from scipy.sparse import coo_matrix 
import numpy as np
import scipy.sparse as sp
import torch

## label the dataset 
#charliehebdo\\rumours\\
#treeDic

def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq=float(pair.split(':')[1])
        index=int(pair.split(':')[0])
        if index<=5409:
            wordFreq.append(freq)
            wordIndex.append(index)
    return wordFreq, wordIndex


def getfeature(x_word,x_index):
    x = np.zeros([len(x_index), 5409]) #doi phu hop voi index_number -> ok
    for i in range(len(x_index)):
        if len(x_index[i])>0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
    return x


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

"""
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
"""

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(pathF, pathT, file_id):
    #path="../data/charliehebdo/rumours/", dataset="rumours" 
    """Load rumors dataset - minh can chac chan cho nay -> ok"""

    #print('Loading {} dataset...'.format(dataset))
    #print('Loading {} dataset...')

    idx_features_labels = np.genfromtxt(pathF+file_id + '.txt',
                                        dtype=np.dtype(str),  delimiter="\t")
    #idx_features_labels = np.genfromtxt("{}{}.features".format(path, dataset),
    #                                    dtype=np.dtype(str)) # +'/'
    
    # build graph # update as saved for fre20 
    idx = np.array(idx_features_labels[:, 2], dtype=np.int64)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    ### update with tree that saved for fre20 
    treeDic = {}
    for line in open(pathF + file_id + '.txt'): # co can phai cong \\?
        line = line.rstrip()
        eid=file_id#'552783238415265792'
        indexC = line.split('\t')[2]#, line.split('\t')[1], int(line.split('\t')[2])
        Vec = line.split('\t')[6]
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'vec': Vec}#{'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
        #print('tree no:', len(treeDic))

    x_word=[]
    x_index=[]
    for j in treeDic[file_id]:
        indexC = j #j o day la cac node id 
        #print(indexC)
        id_node=int(indexC)
        nodeC = idx_map.get(id_node)#index2node[indexC] #nodeC: sau khi da map id cua node id 
        wordFreq, wordIndex = str2matrix(treeDic[file_id][j]['vec'])
        x_word.append(wordFreq)
        x_index.append(wordIndex)
        
    x_feature=getfeature(x_word, x_index)
    
    #features = sp.csr_matrix(x_feature, dtype=np.float32) 
    
    edges_unordered=np.genfromtxt(pathT+file_id + '.txt',
                                        dtype=np.dtype(str),  delimiter="\t")
    
    edges_unordered2=np.array(edges_unordered[:,0:2],dtype=np.int64)
    edges = np.array(list(map(idx_map.get, edges_unordered2.flatten())),
                     dtype=np.int64).reshape(edges_unordered2.shape)
    
    BU_adj = sp.coo_matrix((np.array(edges_unordered[:,2]), (edges[:, 0], edges[:, 1])),
                        shape=(idx.shape[0],idx.shape[0]),
                        dtype=np.float32)
                        
    # adj for bottom up model# sau 20/2 minh se doi lai het TD nhu bigcn, cho nen adj =
    adj=BU_adj.toarray()
    adj=adj.T
    adj=sp.coo_matrix(adj)
    #print(type(BU_adjT))

    # build symmetric adjacency matrix
    #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    #features = normalize(features)
    #adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = normalize(adj)
    BU_adj=normalize(BU_adj)

    #features = torch.FloatTensor(x_feature) #lat them lai 31 January, bo dong duoi nay 
    features = x_feature
    
    #features = torch.FloatTensor(np.array(features.todense())) #old with sparse
    
    #labels = torch.LongTensor(np.where(labels)[1])
    
    ### tam thoi ko xai sparse ma xai darray truoc 
    
    #adj = sparse_mx_to_torch_sparse_tensor(adj)
    #BU_adjT=sparse_mx_to_torch_sparse_tensor(BU_adjT)
    
    adj = adj.toarray()
    BU_adj = BU_adj.toarray()

    return adj, BU_adj, features #, labels
  
data_pathF='E:\\pheme\\datatimeGCN\\features_weightCorrect\\'
data_pathT='E:\\pheme\\datatimeGCN\\delaytime_weightCorrect\\'

adj, BU_adjT, features = load_data(data_pathF, data_pathT, '544521948924248066')
adj
BU_adjT

ider=[]
with open("E:\\pheme\\datatimeGCN\\label4955.txt",'r') as f:
    for line in f:
        lines=line.rstrip()
        file_id=str(lines.split('\t')[0])
        try:
            adj, BU_adj, features = load_data(data_pathF, data_pathT, file_id) #adj here is nparray
            y=int(lines.split('\t')[1])
            np.savez('E:\\pheme\\datatimeGCN\\timegraphCorrect\\'+file_id+'.npz', x=features,adj=adj,BU_adj=BU_adj,y=y) #timegraphuse
        except:
            ider.append(file_id)
            print(file_id)
len(ider) #0

##### ok, loading to see the biggest msize in the graphs 
def loadLabel():
    #if obj == "pheme":
    labelPath = 'E:\\pheme\\datatimeGCN\\label4955.txt'
    #os.path.join(cwd,"data/pheme/label1.txt")
    print("loading label:")
    F, T = [], []
    l1 = l2 = 0
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        eid,label = line.split('\t')[0], line.split('\t')[1]
        labelDic[eid] = int(label)
        if labelDic[eid]==0:
            F.append(eid)
            l1 += 1
        if labelDic[eid]==1:
            T.append(eid)
            l2 += 1
    return labelDic

#data_path = 'E:\\pheme\\datatimeGCN\\timegraphuse\\'
data_path = 'E:\\pheme\\datatimeGCN\\timegraphCorrect\\'
import numpy as np
import pickle 
mt_size=[]
size_dic=dict()
msize = 0
id_in=[]
id_no2=[]

## where I get my labelDic 
labelDic=loadLabel()
for id in labelDic:
  try:
    data = np.load(data_path + id + ".npz", allow_pickle=True)
    x=data['x']
    mt_size.append(int(x.shape[0]))
    maxs = x.shape[0]

    id_in.append(id)
    if msize<maxs:
      msize=maxs

  except:
    id_no2.append(id)
    pass

print("Number of instances:", mt_size)
print("maximu size of the graph:", msize) #77

#524988783213547520
data = np.load("C:\\Users\\s3575040\\My Research\\Reading\\RQ2\\GCN_pheme_rnr\\otta\\ottagraph\\524988783213547520.npz", allow_pickle=True)
data['adj']
##### testing end 

## generating graph data 
data_pathF='./german/features_weight/'
data_pathT='./german/delaytime_weight/'

with open("./german/label1.txt",'r') as f:
    for line in f:
        lines=line.rstrip()
        file_id=lines.split('\t')[0]
        adj, BU_adjT, features = load_data(data_pathF, data_pathT, file_id) #adj here is nparray
        y=int(lines.split('\t')[1])
        np.savez('./german/germangraph/'+file_id+'.npz', x=features,adj=adj,BU_adjT=BU_adjT,y=y)
