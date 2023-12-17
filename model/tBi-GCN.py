##references: https://github.com/tkipf/pygcn; https://github.com/TianBian95 
import json
import time
from torch.utils.data.dataset import random_split
import torch as th
import sys, os
# sys.path.append(os.getcwd())
# device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
from torch_scatter import scatter_mean
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from earlystopping2class import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm

from scipy.sparse import coo_matrix
# import numpy as np
import scipy.sparse as sp

from process.rand5fold import *
from process.evaluate import *
from process.process import *
from layersGCN import GraphConvolution 
import copy
import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data


### define model

class TDrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GraphConvolution(in_feats, hid_feats)  # GCNConv(in_feats, hid_feats)
        self.conv2 = GraphConvolution(hid_feats, out_feats)  # GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        # x, edge_index = data.x, data.edge_index
        x, adj = data.x, data.adj  
        batchsize = max(data.batch) + 1
        x1 = copy.copy(x.float())

        x = x.view(batchsize, 77, 5409)
        adj = adj.view(batchsize, 77, 77)

        x = self.conv1(x, adj)  # edge_index)
        x2 = copy.copy(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)  # should I do this?
        x = self.conv2(x, adj)  # edge_index)

        x = F.relu(x)
        x = x.view(batchsize * 77, 64)
        x = scatter_mean(x, data.batch, dim=0)
        return x

class BUrumorGCN(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GraphConvolution(in_feats, hid_feats)  # GCNConv(in_feats, hid_feats)
        self.conv2 = GraphConvolution(hid_feats, out_feats)  # GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, data):
        x, adj = data.x, data.BUadj
        batchsize = max(data.batch) + 1
        x = x.view(batchsize, 77, 5409)
        adj = adj.view(batchsize, 77, 77)

        x1 = copy.copy(x.float())
        x = self.conv1(x, adj)  # edge_index)
        x2 = copy.copy(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj)  # edge_index)
        x = F.relu(x)
        x = x.view(batchsize * 77, 64)
        x = scatter_mean(x, data.batch, dim=0)
        return x


class Net(th.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.fc = th.nn.Linear((out_feats) * 2, 2) # +hid_feats

    def forward(self, data):
        TD_x = self.TDrumorGCN(data)
        BU_x = self.BUrumorGCN(data)
        x = th.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

############### train, validation, testing ------
## creating some part to save the model and log (for checkinig)
BASE_PATH = 'E:\\pheme\\code\\timegcn\\modelBigcnfold_split3Early'

path_to_fold_saved = BASE_PATH + '\\fold0'  # os.path.join(BASE_PATH,'/fold0')
if os.path.exists(path_to_fold_saved) == False:
    os.makedirs(path_to_fold_saved)

path_to_log = BASE_PATH + '\\logs_fold0'  # os.path.join(BASE_PATH,'/logs_fold0')
if os.path.exists(path_to_log) == False:
    os.makedirs(path_to_log)
path_to_json = os.path.join(path_to_log, 'logs.json')

##save all train and validate and the test results
path_to_jsonResults = os.path.join(path_to_log, 'logs-results.json')

path_to_jsonEarly = os.path.join(path_to_log, 'logs-early.json')


# hyperparameters and load the data: 4 events for training and 1 event for testing
lr = 0.0005
weight_decay = 1e-4
patience=10
n_epochs = 200
batchsize = 128
tddroprate = 0
budroprate = 0
datasetname = "timebigcn"
dataname = "timebigcn"
# iterations= 50
model = "timeBUGCNnoR"

device = th.device('cuda:0')

fold0_x_test, fold0_x_train, \
fold1_x_test, fold1_x_train, \
fold2_x_test, fold2_x_train, \
fold3_x_test, fold3_x_train, \
fold4_x_test, fold4_x_train = load5foldData(datasetname)

treeDic = loadTree()
labelDic = loadLabel()

traindata_list, testdata_list = loadBiData(dataname, treeDic, fold0_x_train, fold0_x_test)

# initial model
model = Net(5409, 64, 64).to(device)  # this include vocab_size, hidden feature for 2 layers
optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


## define train model
def train(trainsub):
    model.train()

    avg_loss, avg_acc = [], []
    batch_idx = 0
    train_loader = DataLoader(trainsub, batch_size=batchsize,
                              shuffle=True, num_workers=0)
    tqdm_train_loader = tqdm(train_loader)
    for Batch_data in tqdm_train_loader:
        Batch_data.to(device)
        out_labels = model(Batch_data)
        loss = F.nll_loss(out_labels, Batch_data.y)
        optimizer.zero_grad()
        loss.backward()
        avg_loss.append(loss.item())
        optimizer.step()
        _, pred = out_labels.max(dim=-1)
        correct = pred.eq(Batch_data.y).sum().item()
        train_acc = correct / len(Batch_data.y)
        avg_acc.append(train_acc)
        postfix = "Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(epoch,
                                                                                                 batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc)
        tqdm_train_loader.set_postfix_str(postfix)
        batch_idx = batch_idx + 1
    epoch_train_loss = np.mean(avg_loss)
    epoch_train_acc = np.mean(avg_acc)

    return epoch_train_loss, epoch_train_acc


##define evaluate model
def test_eval(data_):
    model.eval()

    temp_val_losses, temp_val_accs, temp_val_Acc_all = [], [], []
    temp_val_Acc1, temp_val_Prec1, temp_val_Recll1 = [], [], []
    temp_val_F1, temp_val_Acc2, temp_val_Prec2 = [], [], []
    temp_val_Recll2, temp_val_F2 = [], []

    datatest = DataLoader(data_, batch_size=batchsize,
                          shuffle=True, num_workers=0)

    tqdm_datatest_loader = tqdm(datatest)
    for Batch_data in tqdm_datatest_loader:
        Batch_data.to(device)
        val_out = model(Batch_data)
        val_loss = F.nll_loss(val_out, Batch_data.y)

        temp_val_losses.append(val_loss.item())
        _, val_pred = val_out.max(dim=1)
        correct = val_pred.eq(Batch_data.y).sum().item()
        val_acc = correct / len(Batch_data.y)

        Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluationclass(
            val_pred, Batch_data.y)

        temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
            Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
        temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
            Recll2), temp_val_F2.append(F2)

        temp_val_accs.append(val_acc)

    accs = np.mean(temp_val_Acc_all)
    # acc1 = np.mean(temp_val_Acc1)
    # acc2 = np.mean(temp_val_Acc2)
    pre1 = np.mean(temp_val_Prec1)
    pre2 = np.mean(temp_val_Prec2)
    rec1 = np.mean(temp_val_Recll1)
    rec2 = np.mean(temp_val_Recll2)
    F1 = np.mean(temp_val_F1)
    F2 = np.mean(temp_val_F2)

    epoch_val_losses = np.mean(temp_val_losses)
    epoch_val_accs = np.mean(temp_val_accs)

    return epoch_val_losses, epoch_val_accs, accs, pre1, pre2, rec1, rec2, F1, F2


#### start to train the model
# split the train dataset into training and validationsubset

train_len = int(len(traindata_list) * 0.9)

trainsub, validsub = \
    random_split(traindata_list, [train_len, len(traindata_list) - train_len])

# training and validation with epoch
infor_to_save = {}
infor_early = {}
final_results = {}

train_losses, val_losses, train_accs, val_accs = [], [], [], []  # giong nhu luu lai het moi thu vao 1 list
early_stopping = EarlyStopping(patience=patience, verbose=True)

for epoch in range(n_epochs):

    logs_infor = {'epoch': 0,'epoch_train_loss': [], 'epoch_train_acc': [],'epoch_val_losses': [],
                  'epoch_val_accs': [], 'val_accs_All': [],
                  'val_prec1': [], 'val_prec2': [], 'val_recl1': [], 'val_recl2': [],
                  'val_f1': [], 'val_f2': []}

    logs_infor['epoch'] = epoch

    start_time = time.time()

    epoch_train_loss, epoch_train_acc = train(trainsub)

    logs_infor['epoch_train_loss'].append(round(epoch_train_loss, 5))
    logs_infor['epoch_train_acc'].append(round(epoch_train_acc, 5))

    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)

    epoch_val_losses, epoch_val_accs, accs, pre1, pre2, \
    rec1, rec2, F1, F2 = test_eval(validsub)

    logs_infor['epoch_val_losses'].append(round(epoch_val_losses, 5))
    logs_infor['epoch_val_accs'].append(round(epoch_val_accs, 5))

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
### let's do early stopping (check this function), then
    early_stopping( epoch_val_losses,  accs, pre1,
                   pre2, rec1, rec2,
                   F1, F2, model, 'timeBiGCNnoR_split3fold0', "timeBigcn")
    accs = accs
    #acc1 = np.mean(temp_val_Acc1)
    #acc2 = np.mean(temp_val_Acc2)
    pre1 = pre1
    pre2 = pre2
    rec1 =  rec1
    rec2 = rec2
    F1 = F1
    F2 = F2
    if early_stopping.early_stop:
        print("Early stopping")
        accs = early_stopping.accs
        #acc1 = early_stopping.acc1
        #acc2 = early_stopping.acc2
        pre1 = early_stopping.pre1
        pre2 = early_stopping.pre2
        rec1 = early_stopping.rec1
        rec2 = early_stopping.rec2
        F1 = early_stopping.F1
        F2 = early_stopping.F2

        path_to_save = os.path.join(BASE_PATH + '\\fold0',
                                    f"fold-{0}-epoch-{epoch}-acc-{logs_infor['epoch_val_accs'][-1]:.5f}-loss-{logs_infor['epoch_val_losses'][-1]:.5f}.pth")
        th.save(model.state_dict(), path_to_save)

        infor_early['epoch'] = ['Loss:{:.5}'.format(epoch_val_losses),'acc:{:.5}'.format(accs), 'C1:{:.5f},{:.5f},{:.5f}'.format(pre1,rec1,F1),
                                'C1:{:.5f},{:.5f},{:.5f}'.format(pre2, rec2, F2)]

        with open(path_to_jsonEarly, 'w+') as fE:
            json.dump(infor_early, fE, indent=4)
        break


    # when should I save the model?
    # path_to_save = os.path.join(BASE_PATH + '\\fold0',
    #                             f"fold-{0}-epoch-{epoch}-acc-{logs_infor['epoch_val_accs'][-1]:.5f}-loss-{logs_infor['epoch_val_losses'][-1]:.5f}.pth")
    # th.save(model.state_dict(), path_to_save)

    val_losses.append(round(epoch_val_losses, 5))
    val_accs.append(round(epoch_val_accs, 5))

    logs_infor['val_accs_All'].append(accs)

    logs_infor['val_prec1'].append(pre1)
    logs_infor['val_recl1'].append(rec1)
    logs_infor['val_f1'].append(F1)

    logs_infor['val_prec2'].append(pre2)
    logs_infor['val_recl2'].append(rec2)
    logs_infor['val_f2'].append(F2)

    ## chinh lai cho nay 1 ty
    print('Epoch: %d' % epoch, " | time in %d minutes, %d seconds" % (mins, secs))
    # print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    # print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

    print("Train_Loss {:.5f} | Val_Loss {:.5f}".format(epoch_train_loss, epoch_val_losses))
    print("Train_Accuracy {:.5f}| Val_Accuracy {:.5f}".format(epoch_train_acc, epoch_val_accs))

    res = ['acc:{:.5f}'.format(accs),
           'C1:{:.5f},{:.5f},{:.5f}'.format(pre1, rec1, F1),
           'C2:{:.5f},{:.5f},{:.5f}'.format(pre2, rec2, F2)]
    print('results:', res)

    infor_to_save[str(epoch)] = logs_infor
    with open(path_to_json, 'w+') as f:
        json.dump(infor_to_save, f, indent=4)

print("Finish training")

final_results['train_loss_all'] = train_losses
final_results['train_accs'] = train_accs
final_results['val_losses'] = val_losses
final_results['val_accs'] = val_accs

# evaluate on the test dataset
print('Evaluate time BU model on test dataset...')

testlosses, testaccs, Taccs, Tpre1, Tpre2, Trec1, Trec2, TF1, TF2 = test_eval(testdata_list)

test_results = ['test_losses:{:.5f}'.format(testlosses), 'acc:{:.5f}'.format(Taccs),
                'C1:{:.5f},{:.5f},{:.5f}'.format(Tpre1, Trec1, TF1),
                'C2:{:.5f},{:.5f},{:.5f}'.format(Tpre2, Trec2, TF2)]
print('Test timeBiGCN results on fold0:', test_results)

final_results['test_results'] = test_results

with open(path_to_jsonResults, 'w+') as f1:
    json.dump(final_results, f1, indent=4)
