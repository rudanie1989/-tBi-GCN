# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 11:32:28 2021

@author: S3575040
"""

from scipy.sparse import coo_matrix 
#import numpy as np
import scipy.sparse as sp
import torch

import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import sys

class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.utime=0
        self.parent = None
        
def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq=float(pair.split(':')[1])
        index=int(pair.split(':')[0])
        if index<=5409:
            wordFreq.append(freq)
            wordIndex.append(index)
    return wordFreq, wordIndex

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def constructMat(tree):
    index2node = {}
    for i in tree:
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        utime=float(tree[j]['posttime'])
        
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        nodeC.utime=utime
        
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            #print(indexC)
            rootindex=indexC-1
            root_index=nodeC.index
            root_word=nodeC.word
            root_post=nodeC.utime
            #root_time=nodeC.utime # co nen de cai nay ko? 
            
    rootfeat = np.zeros([1, 5409])
    if len(root_index)>0:
        rootfeat[0, np.array(root_index)] = np.array(root_word)
    adjmatrix=np.zeros([len(index2node),len(index2node)])
    row=[]
    col=[]
    x_word=[]
    x_index=[]
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                if index_i!=index_j:
                    adjmatrix[index_i][index_j]=index2node[index_j+1].utime - index2node[index_i+1].utime #1 #thay doi thanh time, index2node[index_j+1].utime; 
                elif index_i==index_j:
                    adjmatrix[index_i][index_j]=1
                row.append(index_i)
                col.append(index_j)
        x_word.append(index2node[index_i+1].word)
        x_index.append(index2node[index_i+1].index)
    edgematrix=[row,col]
    ## tao luon BU_adj here? hay normalize luon adjmatrix va tao BU_adj o day
    BU_adjT=adjmatrix.T
    BU_adjT=sp.coo_matrix(BU_adjT)
    
    adjmatrix=sp.coo_matrix(adjmatrix)
    
    adjmatrix = normalize(adjmatrix)
    BU_adjT=normalize(BU_adjT)
    
    adjmatrix=adjmatrix.toarray()
    BU_adjT=BU_adjT.toarray()
    
    return x_word, x_index, adjmatrix, BU_adjT, edgematrix,rootfeat,rootindex,root_post #minh muon doi edgematrix thanh matran luon 

#len(index2node)

def getfeature(x_word,x_index):
    x = np.zeros([len(x_index), 2991])
    for i in range(len(x_index)):
        if len(x_index[i])>0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
    return x

def main(): #before obj
    treePath = './german/tree1.txt' # "/content/gdrive/My Drive/Colab Notebooks/GCN/BiGCN/data/pheme/all1.txt"
    print("reading german tree")
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        post_time=line.split('\t')[3]
        Vec = line.split('\t')[4]
        #max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]

        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP,'posttime': post_time, 'vec': Vec} #tree[j]['posttime']
    print('tree no:', len(treeDic))

    labelPath = "./german/label1.txt"
    labelset_R, labelset_nonR =  ['0'], ['1'] #['news', 'non-rumor'], ['false'], ['true'], ['unverified']; labelset_nonR, labelset_f, labelset_t, labelset_u 

    print("loading german label")
    event, y = [], []
    l1 = l2 = 0 #l3 = l4 = 0
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split('\t')[1], line.split('\t')[0]
        label=label.lower()
        event.append(eid)
        if label in labelset_R:
            labelDic[eid]=0
            l1 += 1
        if label  in labelset_nonR:
            labelDic[eid]=1
            l2 += 1
        """
        if label  in labelset_t:
            labelDic[eid]=2
            l3 += 1
        if label  in labelset_u:
            labelDic[eid]=3
            l4 += 1
        """
    print(len(labelDic))
    print(l1, l2)

    def loadEid(event,id,y):
        #id_list=[]
        if event is None:
            return None
        if len(event) < 2:
            return None
        if len(event)>1:
            #print('source tweet:',id)
            try:
                #x_word, x_index, tree, rootfeat, rootindex = constructMat(event)
                x_word, x_index, adjmatrix, BU_adjT, tree,rootfeat,rootindex,root_post = constructMat(event)
                x_x = getfeature(x_word, x_index)
                rootfeat, tree, x_x, rootindex,root_post, adjmatrix, BU_adjT, y = np.array(rootfeat), np.array(tree), np.array(x_x), np.array(
                    rootindex), np.array(root_post), np.array(adjmatrix), np.array(BU_adjT), np.array(y)
                #xem lai cach luu adjmatrix, BU_adjT 
                np.savez('./german/germangraph1/'+id+'.npz', x=x_x,root=rootfeat,edgeindex=tree,rootindex=rootindex,roottime=root_post,TDadj=adjmatrix, BUadj=BU_adjT,y=y)
            except:
                print('source tweet:',id)
                #id_list.append(id)
                pass
            #os.path.join(cwd, 'data/'+obj+'graph/'+id+'.npz'),
            return None
    print("loading dataset", )
    Parallel(n_jobs=1, backend='threading')(delayed(loadEid)(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid]) for eid in tqdm(event))
    return

if __name__ == '__main__':
    #obj= #sys.argv[1]
    main()#main(obj)