import numpy as np
import argparse
import random

from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score
# from VAT import virtual_adversarial_training
# from CRT import CRT, TFR_Encoder, Model
from MAE_T import Model
from base_models import SSLDataSet, FTDataSet

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from subsss import fs_dataset,standardization,normalization
import pandas as pd
from VAT import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Subset


def self_supervised_learning(model, X, n_epoch, lr, batch_size, device, min_ratio=0.3, max_ratio=0.8,num_class = 10):
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)

    model.to(device)
    model.train()

    dataset = SSLDataSet(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    
    pbar = trange(n_epoch)
    for _ in pbar:
        for batch in dataloader:
            x = batch.to(device)
            loss = model(x, ssl=True, ratio=max(min_ratio, min(max_ratio, _ / n_epoch)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        scheduler.step(_)
        pbar.set_description(str(sum(losses) / len(losses)))
    torch.save(model.to('cpu'), f'model_vat/Pretrained_Model_{num_class}.pkl')
    
def finetuning(model, train_set, valid_set, n_epoch, lr, batch_size, device, multi_label=True,n_classes=20,samples_per_class =1, rs=1):
    # multi_label: whether the classification task is a multi-label task.
    model.train()
    model.to(device)
    # 计算要分割的索引
    total_size = len(valid_set)
    valid_set_indices = list(range(total_size))
    train_indices, valid_indices = train_test_split(valid_set_indices, test_size=0.3)

    # 划分数据集
    valid_set_ul = Subset(valid_set, train_indices)
    # print(len(train_indices),len(valid_indices))
    valid_set_l = Subset(valid_set, valid_indices)
    # print(len(valid_set_l))
    print(batch_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader_ul = DataLoader(valid_set_ul, batch_size=batch_size, shuffle=True)
    loss_func = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()
    
    for stage in range(2):
        # stage0: finetuning only classifier; stage1: finetuning whole model
        best_auc = 0
        step = 0
        if stage == 0:
            min_lr = 1e-6
            optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
        else:
            min_lr = 1e-8
            optimizer = optim.Adam(model.parameters(), lr=lr/2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, mode = 'max', factor=0.8, min_lr=min_lr)
        pbar = trange(n_epoch)
        for _ in pbar:
            # for batch_idx, batch in enumerate(train_loader):
            for train_batch, val_batch in zip(train_loader, valid_loader_ul):
                step += 1
                x, y = tuple(t.to(device) for t in train_batch)
                # print(x.shape)
                xul,_ = tuple(t.to(device) for t in val_batch)
                pred = model(x)
                # print(pred.shape, y.shape)
                yul = model(xul)
                # print(pred.shape, yul.shape)
                v_loss = vat_loss(model, xul, yul, eps=2.5)
                loss = loss_func(pred, y)+v_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    valid_load_l = DataLoader(valid_set_l, batch_size=len(valid_set_l), shuffle=False)
                    print(len(valid_set_l))
                    for batch_idx, batch in enumerate(valid_load_l):
                        step += 1
                        valid_X, valid_y = tuple(t.to(device) for t in batch)
                        valid_set_l = FTDataSet(valid_X, valid_y.cpu(), multi_label=False)
                    print(len(valid_set_l))
                    valid_auc = test(model, valid_set_l, batch_size, multi_label)
                    pbar.set_description('Best Validation AUC: {:.4f} --------- AUC on this step: {:.4f},loss{:.4f},vloss{:.4f}'.format(best_auc, valid_auc,loss,v_loss))
                    if valid_auc > best_auc:
                        best_auc = valid_auc
                        torch.save(model, f'model_vat/Finetuned_Model_{n_classes}c{samples_per_class}k_{rs}.pkl')
                    scheduler.step(best_auc)
                    
                    
from sklearn.metrics import accuracy_score                  
def test(model, dataset, batch_size, multi_label):
    model.eval()
    testloader = DataLoader(dataset, batch_size=batch_size)
    # print(len(dataset))
    pred_prob = []
    with torch.no_grad():
        for batch in testloader:
            x, y = tuple(t.to(device) for t in batch)
            pred = model(x)
            pred = torch.sigmoid(pred) if multi_label else F.softmax(pred, dim=1)
            pred_prob.extend([i.cpu().detach().numpy().tolist() for i in pred])
    y_pred = np.argmax(pred_prob, axis=1)
    # Ensure y_pred is moved to CPU before conversion if it's on CUDA
    # if y_pred.is_cuda:
    #     y_pred = y_pred.cpu()

    # It seems your 'dataset.label' might also be a tensor, so apply the same logic
    # if dataset.label.is_cuda:
    #     dataset_label_cpu = dataset.label.cpu()
    # else:
    #     dataset_label_cpu = dataset.label
    # print(y_pred)
    # plot_matrix(dataset.label, y_pred, numclass, samples_per_class)
    print(dataset.label)
    print(y_pred)
    auc = accuracy_score(dataset.label,y_pred)    
    print('AUC is {:.2f}'.format(auc * 100))
    print('More metrics can be added in the test function.')
    model.train()
    return auc

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
        
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import time
from torchsummary import summary
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl", type=str2bool, default=True)
    parser.add_argument("--sl", type=str2bool, default=True)
    parser.add_argument("--load", type=str2bool, default=True)
    parser.add_argument("--test", type=str2bool, default=True)
    # all default values of parameters are for PTB-XL
    parser.add_argument("--seq_len", type=int, default=4800)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--in_dim", type=int, default=2)
    parser.add_argument("--n_classes", type=int, default=20)
    opt = parser.parse_args()
    
    set_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    seq_len = opt.seq_len
    patch_len = opt.patch_len
    dim = opt.dim
    in_dim = opt.in_dim
    n_classes = opt.n_classes
    
    if opt.ssl:
        n_classes =  90
        model = Model(seq_len, patch_len, dim, n_classes, in_dim).to(device)
        X = np.load('Dataset_4800/X_train_90Class.npy').transpose(0, 2, 1)
        # print(X.max())
        X,_,_ = normalization(X)
        # print(X.max())
        # Y = 'Dataset_4800/Y_train_90Class.npy'
        # train_set = MyDataset(X,Y)
        start = time.time()
        self_supervised_learning(model, X, 100, 1e-3, 128, device, num_class = n_classes)
        end = time.time()
        ssl_time = end-start
    # change here!!!!
    Ks = [20]#1, 5, 10, 15, #shots 
    num_Ks = np.shape(Ks)[0]
    Ns = [10] #10, 20, 30 #classes
    num_Ns = np.shape(Ns)[0]
    Rs = 1 #experiment number
    acc = np.zeros([int(num_Ks*num_Ns), Rs])
    index_ = [f'{i}c{j}k' for i in Ns for j in Ks]   
    columns_ = [f'{i}' for i in range(Rs) ]
    # df2 = pd.DataFrame(acc, 
    #         index = index_, 
    #         columns = columns_)
    for r in range(Rs):
        for n in range(num_Ns):
            for k in range(num_Ks):
                n_classes = Ns[n]      
                samples_per_class = Ks[k]
                if opt.load:
                    model = Model(seq_len, patch_len, dim, n_classes, in_dim).to(device)
                    pre_weights = torch.load(f'model_vat/Pretrained_Model_90.pkl', map_location=device)
                    # print(pre_weights.state_dict().items())
                    pre_dict = {k:v for k,v in pre_weights.state_dict().items() if "classifier" not in k}
                    missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)
                else:
                    model = Model(seq_len, patch_len, dim, n_classes, in_dim).to(device)

                if opt.sl:
                    data_path = f'Dataset_4800/X_train_{n_classes}Class.npy'
                    label_path = f'Dataset_4800/Y_train_{n_classes}Class.npy'
                    train_X, train_y, valid_X, valid_y = fs_dataset(data_path, label_path,samples_per_class = samples_per_class, num_classes =n_classes)
                    # print(train_X.shape)
                    min_value = np.minimum(np.min(train_X), np.min(valid_X))
                    max_value = np.maximum(np.max(train_X), np.max(valid_X))
                    train_X = (train_X - min_value) / (max_value - min_value)
                    valid_X = (valid_X - min_value) / (max_value - min_value)
                    TrainSet = FTDataSet(train_X, train_y, multi_label=False) # multi_label = True when the dataset is PTBXL
                    ValidSet = FTDataSet(valid_X, valid_y, multi_label=False)
                    start = time.time()
                    finetuning(model, TrainSet, ValidSet, 100, 1e-3, 16, device, multi_label=False,n_classes=n_classes,samples_per_class = samples_per_class, rs=r)
                    end = time.time()
                    ft_time = end-start
                if opt.test:
                    test_X, test_y = np.load(f'Dataset_4800/X_test_{n_classes}Class.npy').transpose(0, 2, 1), np.load(f'Dataset_4800/Y_test_{n_classes}Class.npy')
                    test_X = (test_X - min_value) / (max_value - min_value)
                    TestSet = FTDataSet(test_X, test_y, multi_label=False)
                    # model = Model(seq_len, patch_len, dim, n_classes, in_dim).to(device)
                    # model = torch.load(f'Finetuned_Model_{n_classes}c{samples_per_class}k_{r}.pkl', map_location=device)
                    model = torch.load(f'model_vat/Finetuned_Model_{n_classes}c{samples_per_class}k_{r}.pkl', map_location=device)
                    start = time.time()
                    auc = test(model, TestSet, 100, multi_label=False)
                    end = time.time()
                    test_time = end-start
                acc[n*num_Ks + k, r] = auc
                df2 = pd.DataFrame(acc, 
                        index = index_, 
                        columns = columns_)
                # df2.to_csv("result/DCTAE_all.csv")
                df2.to_csv("DCTAE_all.csv")
    acc_3m = np.zeros([int(num_Ks*num_Ns), 3])
    acc_3m[:, 0] = np.mean(acc, axis = 1)
    acc_3m[:, 1] = np.max(acc, axis = 1)
    acc_3m[:, 2] = np.min(acc, axis = 1)
#0.9
    df = pd.DataFrame(acc_3m, index = index_, columns=['mean','max', 'min'])
    # df.to_csv("result/DCTAE_3m.csv")
    df.to_csv("DCTAE_3m.csv")
    print(f'SSL time: {ssl_time}，FT time: {ft_time}，Test time: {test_time}')
    # if opt.load:
    #     model = torch.load(f'Pretrained_Model_{n_classes}.pkl', map_location=device)
    # else:
    #     model = Model(seq_len, patch_len, dim, n_classes, in_dim).to(device)
    # if opt.sl:
    #     data_path = f'Dataset_4800/X_train_{n_classes}Class.npy'
    #     label_path = f'Dataset_4800/Y_train_{n_classes}Class.npy'
    #     train_X, train_y, valid_X, valid_y = fs_dataset(data_path, label_path,samples_per_class = samples_per_class, num_classes =n_classes)
    #     TrainSet = FTDataSet(train_X, train_y, multi_label=False) # multi_label = True when the dataset is PTBXL
    #     ValidSet = FTDataSet(valid_X, valid_y, multi_label=False)
    #     finetuning(model, TrainSet, ValidSet, 100, 1e-3, 128, device, multi_label=False)
    # if opt.test:
    #     test_X, test_y = np.load(f'Dataset_4800/X_test_{n_classes}Class_train5val95.npy'), np.load(f'Dataset_4800/Y_test_{n_classes}Class.npy')
    #     TestSet = FTDataSet(test_X, test_y, multi_label=False)
    #     model = torch.load('Finetuned_Model.pkl', map_location=device)
    #     auc = test(model, TestSet, 100, multi_label=False)
        
    
