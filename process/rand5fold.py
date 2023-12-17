# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:00:29 2021

@author: S3575040
"""

import random
from random import shuffle
import os

cwd = os.getcwd()


def load5foldData(obj):

    if obj == "timebigcn":
        labelPath = 'E:\\pheme\\datatimeGCN\\label4955.txt'
        cha, fer, ger, otta, syd = [], [], [], [], []
        l1 = l2 = l3 = l4 = l5 = 0
        labelDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            eid, label, datalabel = line.split('\t')[0], line.split('\t')[1], line.split('\t')[2]
            labelDic[eid] = int(label)
            if datalabel == 'chalie':
                cha.append(eid)
                l1 += 1
            if datalabel == 'ferguson':
                fer.append(eid)
                l2 += 1
            if datalabel == 'german':
                ger.append(eid)
                l3 += 1
            if datalabel == 'otta':
                otta.append(eid)
                l4 += 1
            if datalabel == 'syd':
                syd.append(eid)
                l5 += 1

        print(len(labelDic))
        print(l1, l2, l3, l4, l5)
        random.shuffle(cha)
        random.shuffle(fer)
        random.shuffle(ger)
        random.shuffle(otta)
        random.shuffle(syd)

        fold0_x_test, fold1_x_test, fold2_x_test, fold3_x_test, fold4_x_test = [], [], [], [], []
        fold0_x_train, fold1_x_train, fold2_x_train, fold3_x_train, fold4_x_train = [], [], [], [], []
        fold0_x_test.extend(syd)
        fold0_x_train.extend(cha)
        fold0_x_train.extend(fer)
        fold0_x_train.extend(ger)
        fold0_x_train.extend(otta)

        fold1_x_test.extend(otta)
        fold1_x_train.extend(cha)
        fold1_x_train.extend(fer)
        fold1_x_train.extend(ger)
        fold1_x_train.extend(syd)

        fold2_x_test.extend(ger)
        fold2_x_train.extend(cha)
        fold2_x_train.extend(fer)
        fold2_x_train.extend(syd)
        fold2_x_train.extend(otta)

        fold3_x_test.extend(fer)
        fold3_x_train.extend(cha)
        fold3_x_train.extend(syd)
        fold3_x_train.extend(ger)
        fold3_x_train.extend(otta)

        fold4_x_test.extend(cha)
        fold4_x_train.extend(fer)
        fold4_x_train.extend(syd)
        fold4_x_train.extend(ger)
        fold4_x_train.extend(otta)

    fold0_test = list(fold0_x_test)
    shuffle(fold0_test)
    len(fold0_test)
    fold0_train = list(fold0_x_train)
    shuffle(fold0_train)
    len(fold0_train)

    fold1_test = list(fold1_x_test)
    shuffle(fold1_test)
    fold1_train = list(fold1_x_train)
    shuffle(fold1_train)
    fold2_test = list(fold2_x_test)
    shuffle(fold2_test)
    fold2_train = list(fold2_x_train)
    shuffle(fold2_train)
    fold3_test = list(fold3_x_test)
    shuffle(fold3_test)
    fold3_train = list(fold3_x_train)
    shuffle(fold3_train)
    fold4_test = list(fold4_x_test)
    shuffle(fold4_test)
    fold4_train = list(fold4_x_train)
    shuffle(fold4_train)

    # res = [[list(fold0_test), list(fold0_train)],
    #        [list(fold1_test), list(fold1_train)],
    #        [list(fold2_test), list(fold2_train)],
    #        [list(fold3_test), list(fold3_train)],
    #        [list(fold4_test), list(fold4_train)]]
    return list(fold0_test),list(fold0_train),\
           list(fold1_test),list(fold1_train),\
           list(fold2_test),list(fold2_train),\
           list(fold3_test),list(fold3_train),\
           list(fold4_test), list(fold4_train)
