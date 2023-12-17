def loadTree():  # can we change "dataname" into "path"
    # if 'timebigcn' in dataname:
    treePath = 'E:\\pheme\\data_Bigcn\\tree_GCN.txt'
    print("reading tree")
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], line.split('\t')[2]
        Vec = line.split('\t')[4]
        # eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        # max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
        # treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
    return treeDic


def loadLabel():
    # if obj == "pheme":
    labelPath = 'E:\\pheme\\datatimeGCN\\label4955.txt'  # './' + dataname + '/label1.txt'
    # os.path.join(cwd,"data/pheme/label1.txt")
    print("loading label:")
    F, T = [], []
    l1 = l2 = 0
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        eid, label = line.split('\t')[0], line.split('\t')[1]
        labelDic[eid] = int(label)
        if labelDic[eid] == 0:
            F.append(eid)
            l1 += 1
        if labelDic[eid] == 1:
            T.append(eid)
            l2 += 1
    return labelDic


msize = 77


class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000,
                 data_path='E:\\pheme\\datatimeGCN\\timegraphCorrect\\'):
        self.fold_x = list(
            filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        # self.labelDic = labelDic
        self.data_path = data_path
        # self.data_pathF = data_pathF
        # self.data_pathT = data_pathT
        # self.data_label = data_label
        # self.tddroprate = tddroprate
        # self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    # i add this

    def __getitem__(self, index):
        id = self.fold_x[index]
        # each id, I will load the input to GCN model and the label
        # _, _, features = load_data(self.data_pathF, self.data_pathT, id)
        data = np.load(self.data_path + id + ".npz", allow_pickle=True)
        # adj, BU_adjT, features = load_data(self.data_pathF, self.data_pathT, id)
        adj = data['adj']
        BUadj = data['BU_adj']
        x = data['x']
        new_adj = np.pad(adj, [(0, msize - adj.shape[0]), (0, msize - adj.shape[0])], mode='constant')
        new_BUadj = np.pad(BUadj, [(0, msize - BUadj.shape[0]), (0, msize - BUadj.shape[0])], mode='constant')

        new_x = np.pad(x, [(0, msize - adj.shape[0]), (0, 0)], mode='constant')
        # adj=adj.reshape((adj.shape[0]*adj.shape[1],1))
        # features=data['features']
        y = data['y']

        # get the label here
        # y=self.labelDic[id]
        # print(y)
        # print(data['adj'],)
        return Data(x=torch.tensor(new_x, dtype=torch.float32), adj=torch.tensor(new_adj, dtype=torch.float32),
                    BUadj=torch.tensor(new_BUadj, dtype=torch.float32),
                    y=torch.LongTensor([int(
                        y)]))  # TD_adj=torch.tensor(adj,dtype=torch.float32), BU_adj=torch.tensor(BU_adjT,dtype=torch.float32),


def collate_fn(data):
    return data


def loadBiData(dataname, treeDic, fold_x_train, fold_x_test):
    data_path = 'E:\\pheme\\datatimeGCN\\timegraphCorrect\\'

    print("loading train set", )
    traindata_list = BiGraphDataset(fold_x_train, treeDic,
                                    data_path=data_path)  # labelDic, F=data_pathF,data_pathT=data_pathT)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = BiGraphDataset(fold_x_test, treeDic,
                                   data_path=data_path)  # labelDic, data_pathF=data_pathF,data_pathT=data_pathT)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list


