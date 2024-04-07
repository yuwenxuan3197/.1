import numpy as np
from numpy import random
import pickle
from scipy.sparse import csr_matrix
import math
import gc
import time
import random
import datetime
import multiprocessing
import torch as t
import torch.nn as nn
import torch.utils.data as dataloader
import torch.nn.functional as F
from torch.nn import init
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import graph_utils
import DataHandler
import warnings
import torch
warnings.filterwarnings("ignore", category=DeprecationWarning)
cores = multiprocessing.cpu_count() // 2
import BGNN
import BGNN11
import MV_Net
from sklearn.metrics import roc_auc_score
from Params import args
from Utils.TimeLogger import log
from tqdm import tqdm
import scipy.sparse as sp
t.backends.cudnn.benchmark = True
from utility.optimize import HMG
if t.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False

MAX_FLAG = 0x7FFFFFFF

now_time = datetime.datetime.now()
modelTime = datetime.datetime.strftime(now_time, '%Y_%m_%d__%H_%M_%S')

t.autograd.set_detect_anomaly(True)


class Model():
    def __init__(self):

        self.trn_file = args.path + args.dataset + '/trn_'
        self.tst_file = args.path + args.dataset + '/tst_int'
        # self.tst_file = args.path + args.dataset + '/BST_tst_int_59'
        # Tmall: 3,4,5,6,8,59
        # IJCAI_15: 5,6,8,10,13,53

        # self.meta_multi_file = args.path + args.dataset + '/meta_multi_beh_user_index'
        self.t_max = -1
        self.t_min = 0x7FFFFFFF
        self.time_number = -1

        self.user_num = -1
        self.item_num = -1
        self.behavior_mats = {}
        self.behaviors = []
        self.behaviors_data = {}

        # history
        self.train_loss = []
        self.train_loss2 = []
        self.his_hr = []
        self.his_ndcg = []
        gc.collect()  #
        self.relu = t.nn.ReLU()
        self.sigmoid = t.nn.Sigmoid()
        self.curEpoch = 0
        self.tstDicts = dict()
        self.trnDicts = [dict()]

        if args.dataset == 'Tmall':
            self.behaviors_SSL = ['pv', 'fav', 'cart', 'buy']
            self.behaviors = ['pv', 'fav', 'cart', 'buy']
            # self.behaviors = ['buy']
        elif args.dataset == 'IJCAI_15':
            self.behaviors = ['click', 'fav', 'cart', 'buy']
            # self.behaviors = ['buy']
            self.behaviors_SSL = ['click', 'fav', 'cart', 'buy']

        elif args.dataset == 'JD':
            self.behaviors = ['review', 'browse', 'buy']
            self.behaviors_SSL = ['review', 'browse', 'buy']
        elif args.dataset == "Taobao":
            self.behaviors = ['pv', 'cart', 'buy']
            self.behaviors_SSL = ['pv', 'cart', 'buy']
        elif args.dataset == "Beibei":
            self.behaviors = ['pv', 'cart', 'buy']
            self.behaviors_SSL = ['pv', 'cart', 'buy']
        elif args.dataset == 'retailrocket':
            self.behaviors = ['view', 'cart', 'buy']
            # self.behaviors = ['buy']
            self.behaviors_SSL = ['view', 'cart', 'buy']
        # print(self.trn_file + self.behaviors[0])
        with open(args.path + args.dataset + '/train.txt', 'r', encoding='utf-8', errors='ignore') as f:
            for l in f.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, train_items = items[0], items[1:]
                # self.interNum[-1] += len(train_items)
                self.trnDicts[-1][uid] = train_items
        with open(args.path + args.dataset + '/test.txt', 'r', encoding='utf-8', errors='ignore') as f:
            for l in f.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                try:
                    items = [int(i) for i in l.split(' ')]
                except Exception:
                    continue
                uid, test_items = items[0], items[1:]
                self.tstDicts[uid] = test_items
        for i in range(0, len(self.behaviors)):
            with open(self.trn_file + self.behaviors[i], 'rb') as fs:
                data = pickle.load(fs)
                self.behaviors_data[i] = data

                if data.get_shape()[0] > self.user_num:
                    self.user_num = data.get_shape()[0]
                if data.get_shape()[1] > self.item_num:
                    self.item_num = data.get_shape()[1]

                if data.data.max() > self.t_max:
                    self.t_max = data.data.max()
                if data.data.min() < self.t_min:
                    self.t_min = data.data.min()

                if self.behaviors[i] == args.target:
                    self.trainMat = data
                    self.trainLabel = 1 * (self.trainMat != 0)
                    self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0)))

        #        self.behaviors_data[4] = (self.behaviors_data[3] != 0).multiply(self.behaviors_data[0])
        #       self.behaviors_data[5] = (self.behaviors_data[3] != 0).multiply(self.behaviors_data[1])
        #      self.behaviors_data[6] = (self.behaviors_data[3] != 0).multiply(self.behaviors_data[2])
        #     self.behaviors_data[7] = (self.behaviors_data[0] != 0).multiply(self.behaviors_data[1])
        #     self.behaviors_data[8] = (self.behaviors_data[0] != 0).multiply(self.behaviors_data[2])
        #    self.behaviors_data[9] = (self.behaviors_data[1] != 0).multiply(self.behaviors_data[2])
        time = datetime.datetime.now()

        self.trnMats = [sp.dok_matrix((self.user_num, self.item_num), dtype=np.float32) for i in
                   range(len(self.behaviors))]

        for i in range(len(self.behaviors)):
            beh = self.behaviors[i]
            beh_filename = args.path + args.dataset + '/' + beh + '.txt'
            with open(beh_filename) as f:
                for l in f.readlines():
                    if len(l) == 0: break
                    l = l.strip('\n')
                    items = [int(i) for i in l.split(' ')]
                    uid, items_list = items[0], items[1:]
                    for item in items_list:
                        self.trnMats[i][uid, item] = 1.
        """beh_filename = args.path + args.dataset + '/' + "total" + '.txt'
        with open(beh_filename) as f:
            for l in f.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                items = [int(i) for i in l.split(' ')]
                uid, items_list = items[0], items[1:]
                for item in items_list:
                    self.trnMats[4][uid, item] = 1."""
        self.R0=self.create_adj_mat(self.trnMats[0])
        self.R1=self.create_adj_mat(self.trnMats[1])
        self.R2=self.create_adj_mat(self.trnMats[2])
        self.R3=self.create_adj_mat(self.trnMats[3])
        #self.R4=self.create_adj_mat(self.trnMats[4])

        self.all_h_list0, self.all_t_list0, self.all_v_list0 = self.load_adjacency_list_data(self.R0)
        self.all_h_list1, self.all_t_list1, self.all_v_list1 = self.load_adjacency_list_data(self.R1)
        self.all_h_list2, self.all_t_list2, self.all_v_list2 = self.load_adjacency_list_data(self.R2)
        self.all_h_list3, self.all_t_list3, self.all_v_list3 = self.load_adjacency_list_data(self.R3)


        print("Start building:  ", time)
        for i in range(0, len(self.behaviors)):
            self.behavior_mats[i] = graph_utils.get_use(self.behaviors_data[i])
        time = datetime.datetime.now()
        print("End building:", time)

        #print("user_num: ", self.user_num)
        #print("item_num: ", self.item_num)
        print("\n")

        # ---------------------------------------------------------------------------------------------->>>>>
        # train_data
        train_u, train_v = self.trainMat.nonzero()
        train_data = np.hstack((train_u.reshape(-1, 1), train_v.reshape(-1, 1))).tolist()
        train_dataset = DataHandler.RecDataset_beh(self.behaviors, train_data, self.item_num, self.behaviors_data, True)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4,
                                                  pin_memory=True)

        # valid_data

        # test_data
        with open(self.tst_file, 'rb') as fs:
            data = pickle.load(fs)

        test_user = np.array([idx for idx, i in enumerate(data) if i is not None])
        test_item = np.array([i for idx, i in enumerate(data) if i is not None])
        # tstUsrs = np.reshape(np.argwhere(data!=None), [-1])
        test_data = np.hstack((test_user.reshape(-1, 1), test_item.reshape(-1, 1))).tolist()
        # testbatch = np.maximum(1, args.batch * args.sampNum
        test_dataset = DataHandler.RecDataset(test_data, self.item_num, self.trainMat, 0, False)
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=4,
                                                 pin_memory=True)
        # -------------------------------------------------------------------------------------------------->>>>>

    def load_adjacency_list_data(self,adj_mat):
        tmp = adj_mat.tocoo()
        all_h_list = list(tmp.row)
        all_t_list = list(tmp.col)
        all_v_list = list(tmp.data)

        return all_h_list, all_t_list, all_v_list
    def create_adj_mat(self, which_R):
        adj_mat = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)  # 全图邻接矩阵
        adj_mat = adj_mat.tolil()
        R = which_R.tolil()
        adj_mat[:self.user_num, self.user_num:] = R
        adj_mat[self.user_num:, :self.user_num] = R.T
        adj_mat = adj_mat.todok()
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat_inv)
        print('generate pre adjacency matrix.')
        pre_adj_mat = norm_adj.tocsr()
        return pre_adj_mat.tocsr()

    def prepareModel(self):
        self.modelName = self.getModelName()
        self.setRandomSeed()
        self.gnn_layer = eval(args.gnn_layer)  # [16,16,16]
        self.hidden_dim = args.hidden_dim  # 16

        if args.isload == True:
            self.loadModel(args.loadModelPath)
        else:
            self.model = BGNN.myModel(self.user_num, self.item_num, self.behaviors, self.behavior_mats,self.R0,self.R1,self.R2,self.R3,self.R0,self.all_h_list0,self.all_h_list1,self.all_h_list2,self.all_h_list3,self.all_t_list0,self.all_t_list1,self.all_t_list2,self.all_t_list3).cuda()
            self.meta_weight_net = MV_Net.MetaWeightNet(len(self.behaviors)).cuda()
            self.model1=BGNN11.myModel(self.user_num, self.item_num, self.behaviors, self.behavior_mats,self.R0,self.R1,self.R2,self.R3,self.all_h_list0,self.all_h_list1,self.all_h_list2,self.all_h_list3,self.all_t_list0,self.all_t_list1,self.all_t_list2,self.all_t_list3).cuda()

        # #IJCAI_15
        # self.opt = t.optim.AdamW(self.model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
        # self.meta_opt =  t.optim.AdamW(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay)
        # # self.meta_opt =  t.optim.RMSprop(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay, momentum=0.95, centered=True)
        # self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=5, step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        # self.meta_scheduler = t.optim.lr_scheduler.CyclicLR(self.meta_opt, args.meta_opt_base_lr, args.meta_opt_max_lr, step_size_up=2, step_size_down=3, mode='triangular', gamma=0.98, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.9, max_momentum=0.99, last_epoch=-1)
        # #

        # Tmall
        self.opt1 = torch.optim.Adam(self.model1.parameters(), lr=0.01)
        self.opt = t.optim.AdamW(self.model.parameters(), lr=args.lr)
        # self.meta_opt =  t.optim.RMSprop(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay, momentum=0.95, centered=True)
        self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=5,
                                                       step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None,
                                                       scale_mode='cycle', cycle_momentum=False, base_momentum=0.8,
                                                       max_momentum=0.9, last_epoch=-1)


        #                                                                                                                                                                           0.993

        # # retailrocket
        # self.opt = t.optim.AdamW(self.model.parameters(), lr = args.lr, weight_decay = args.opt_weight_decay)
        # # self.meta_opt =  t.optim.AdamW(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay)
        # self.meta_opt =  t.optim.SGD(self.meta_weight_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_opt_weight_decay, momentum=0.95, nesterov=True)
        # self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=1, step_size_down=2, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        # self.meta_scheduler = t.optim.lr_scheduler.CyclicLR(self.meta_opt, args.meta_opt_base_lr, args.meta_opt_max_lr, step_size_up=1, step_size_down=2, mode='triangular', gamma=0.99, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.9, max_momentum=0.99, last_epoch=-1)
        # #                                                                                                                                      exp_range

        if use_cuda:
            self.model = self.model.cuda()

    def innerProduct(self, u, i, j):
        pred_i = t.sum(t.mul(u, i), dim=1) * args.inner_product_mult
        pred_j = t.sum(t.mul(u, j), dim=1) * args.inner_product_mult
        return pred_i, pred_j

    def SSL1(self, user_embeddings, user_step_index,indics):

        def multi_neg_sample_pair_index(batch_index, step_index, embedding1,
                                        embedding2):

            index_set = set(np.array(step_index.cpu()))
            batch_index_set = set(np.array(batch_index.cpu()))
            neg2_index_set = index_set - batch_index_set
            neg2_index = t.as_tensor(np.array(list(neg2_index_set))).long().cuda()
            neg2_index = t.unsqueeze(neg2_index, 0)
            neg2_index = neg2_index.repeat(len(batch_index), 1)
            neg2_index = t.reshape(neg2_index, (1, -1))
            neg2_index = t.squeeze(neg2_index)

            neg1_index = batch_index.long().cuda()
            neg1_index = t.unsqueeze(neg1_index, 1)
            neg1_index = neg1_index.repeat(1, len(neg2_index_set))
            neg1_index = t.reshape(neg1_index, (1, -1))
            neg1_index = t.squeeze(neg1_index)

            neg_score_pre = t.sum(
                compute(embedding1, embedding2, neg1_index, neg2_index).squeeze().view(len(batch_index), -1),
                -1)
            return neg_score_pre

        def compute(x1, x2, indics,neg1_index=None, neg2_index=None, τ=0.05):

            if neg1_index != None:
                x1 = x1[neg1_index]
                x2 = x2[neg2_index]

            N = x1.shape[0]
            D = x1.shape[1]
            #target_graph=self.behavior_mats[3]["A"]
            #indis=torch.topk((target_graph@target_graph.T).fill_diagonal_(0),4)[0]
            x1 = x1
            x2 = x2
            import os
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 下面老是报错 shape 不一致

            a = F.normalize(x1)
            b = F.normalize(x2)
            #indis=torch.topk((a@a.T).fill_diagonal_(0),3)[1]
            #values=torch.topk(a@b.T,3)[0]
            #print((a @ b.T).shape)
            

            
            """values=(a @ b.T).gather(1, indics)
            pos_val=torch.diag(a @ b.T)
            po=t.exp((values.sum(dim=1))/0.3)
            ttl_score_user=(a @ b.T).fill_diagonal_(0)
            mask = torch.ones_like(ttl_score_user)
            for i in range(indics.shape[0]):
                mask[i, indics[i]] = 0
            c=torch.masked_select(ttl_score_user, mask.bool())
            c = c.view(ttl_score_user.shape[0], ttl_score_user.shape[1] - indics.shape[1])
            c=torch.exp(c / 0.3)
            #kk=((a @ b.T).sum(dim=1) - values.sum(dim=1) - pos_val)
            #kk/0.1
            neg=c.sum(dim=1)
            #scores = t.exp(t.div(t.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1), np.power(D, 1) + 1e-8))
            scores=-torch.log(po / neg).sum()"""
            
            indics=indics.cuda()
            values=(a @ b.T).gather(1, indics)[:,1]
            pos_val=torch.diag(a @ b.T)
            po=t.exp((pos_val*0.65+values*0.45)/0.2)
            
            """ttl_score_user=(a @ b.T).fill_diagonal_(0)
            mask = torch.ones_like(ttl_score_user)
            for i in range(indics.shape[0]):
                #mask[i, indics[i][0]] = 0
                mask[i, indics[i][1]] = 0
            c=torch.masked_select(ttl_score_user, mask.bool())
            #print(c)
            c = c.view(ttl_score_user.shape[0], ttl_score_user.shape[1] - indics.shape[1]+1)"""
            
            c=a@b.T
            
            c=torch.exp(c / 0.2)
            #kk=((a @ b.T).sum(dim=1) - values.sum(dim=1) - pos_val)
            #kk/0.1
            neg=c.sum(dim=1)
            #scores = t.exp(t.div(t.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1), np.power(D, 1) + 1e-8))
            scores=-torch.log(po / neg).sum()            
            
            
            
            
            return scores

        def single_infoNCE_loss_one_by_one(embedding1, embedding2, step_index,indics):
            N = step_index.shape[0]
            D = embedding1.shape[1]
            #embedding1[step_index]
            pos_score = compute(embedding1[step_index], embedding2[step_index],indics)
            return pos_score
        def single_infoNCE_loss_one_by_one1(embedding1, embedding2, step_index,indics):
            N = step_index.shape[0]
            D = embedding1.shape[1]
            a = F.normalize(embedding1[step_index])
            b = F.normalize(embedding2[step_index])
            pos_val=torch.diag(a @ b.T)
            po=t.exp((pos_val)/0.2)

            ttl_score = torch.matmul(a, b.transpose(0, 1))
            ttl_score = torch.exp(ttl_score / 0.2).sum(dim=1)  
            scores=-torch.log(po / ttl_score).sum()
            #embedding1[step_index]
            #pos_score = compute(embedding1[step_index], embedding2[step_index],indics)
            return scores


        user_con_loss_list = []

        for i in range(len(self.behaviors_SSL)):
            if i==1:
                user_con_loss_list.append(single_infoNCE_loss_one_by_one1(user_embeddings[-1], user_embeddings[i], user_step_index,indics))
            if i==2:
                user_con_loss_list.append(single_infoNCE_loss_one_by_one1(user_embeddings[-1], user_embeddings[i], user_step_index,indics))
            #user_con_loss_list.append(single_infoNCE_loss_one_by_one1(user_embeddings[-1], user_embeddings[i], user_step_index,indics))
            user_con_loss_list.append(single_infoNCE_loss_one_by_one(user_embeddings[-1], user_embeddings[i], user_step_index,indics))
        return user_con_loss_list, user_step_index

    def run(self):

        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        cur_best_pre_0 = 0
        stopping_step = 0
        self.prepareModel()
        if args.isload == True:
            print("----------------------pre test:")
            HR, NDCG = self.testEpoch(self.test_loader)
            print(f"HR: {HR} , NDCG: {NDCG}")
        log('Model Prepared')

        cvWait = 0
        self.best_HR = 0
        self.best_NDCG = 0
        flag = 0
        self.nonshared_idx=-1
        self.nonshared_idx1 = -1
        self.user_embed = None
        self.item_embed = None
        self.user_embeds = None
        self.item_embeds = None

        print("Test before train:")
        # ret= self.testEpoch(self.test_loader)

        for e in range(self.curEpoch, args.epoch + 1):
            self.curEpoch = e

            self.meta_flag = 0
            if e % args.meta_slot == 0:
                self.meta_flag = 1
            log("*****************Start epoch: %d ************************" % e)

            if args.isJustTest == False:
                epoch_loss, user_embed, item_embed, user_embeds, item_embeds = self.trainEpoch()
                self.train_loss.append(epoch_loss)
                print(f"epoch {e / args.epoch},  epoch loss{epoch_loss}")
                self.train_loss.append(epoch_loss)

            else:
                break

            HR, NDCG = self.testEpoch(self.test_loader)
            self.his_hr.append(HR)
            self.his_ndcg.append(NDCG)         

            self.scheduler.step()

            if HR > self.best_HR:
                self.best_HR = HR
                self.best_epoch = self.curEpoch
                cvWait = 0
                print(
                    "--------------------------------------------------------------------------------------------------------------------------best_HR",
                    self.best_HR)
                # print("--------------------------------------------------------------------------------------------------------------------------NDCG", self.best_NDCG)
                #self.user_embed = user_embed
                self.item_embed = item_embed
                self.user_embeds = user_embeds
                self.item_embeds = item_embeds

                # self.saveHistory()
                # self.saveModel()

            if NDCG > self.best_NDCG:
                self.best_NDCG = NDCG
                self.best_epoch = self.curEpoch
                cvWait = 0
                # print("--------------------------------------------------------------------------------------------------------------------------HR", self.best_HR)
                print(
                    "--------------------------------------------------------------------------------------------------------------------------best_NDCG",
                    self.best_NDCG)
                #self.user_embed = user_embed
                self.item_embed = item_embed
                self.user_embeds = user_embeds
                self.item_embeds = item_embeds
 
                # self.saveHistory()
                # self.saveModel()

            if (HR < self.best_HR) and (NDCG < self.best_NDCG):
                cvWait += 1

            if cvWait == args.patience:
                print(f"Early stop at {self.best_epoch} :  best HR: {self.best_HR}, best_NDCG: {self.best_NDCG} \n")
                # self.saveHistory()
                # self.saveModel()
                break

        HR, NDCG = self.testEpoch(self.test_loader)
        self.his_hr.append(HR)
        self.his_ndcg.append(NDCG)

    def early_stopping_new(self, log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
        update_flag = False
        # early stopping strategy:
        assert expected_order in ['acc', 'dec']

        if (expected_order == 'acc' and log_value >= best_value) or (
                expected_order == 'dec' and log_value <= best_value):
            stopping_step = 0
            best_value = log_value
            update_flag = True  # find a better model
        else:
            stopping_step += 1

        if stopping_step >= flag_step:
            print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
            should_stop = True
        else:
            should_stop = False
        return best_value, stopping_step, should_stop, update_flag

    def negSamp(self, temLabel, sampSize, nodeNum):
        negset = [None] * sampSize
        cur = 0
        while cur < sampSize:
            rdmItm = np.random.choice(nodeNum)
            if temLabel[rdmItm] == 0:
                negset[cur] = rdmItm
                cur += 1
        return negset

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds.cpu()].toarray()
        batch = len(batIds)
        user_id = []
        item_id_pos = []
        item_id_neg = []

        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(labelMat.shape[1])]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = self.negSamp(temLabel[i], sampNum, labelMat.shape[1])

            for j in range(sampNum):
                user_id.append(batIds[i].item())
                item_id_pos.append(poslocs[j].item())
                item_id_neg.append(neglocs[j])
                cur += 1

        return t.as_tensor(np.array(user_id)).cuda(), t.as_tensor(np.array(item_id_pos)).cuda(), t.as_tensor(
            np.array(item_id_neg)).cuda()

    def trainEpoch(self):
        train_loader = self.train_loader
        time = datetime.datetime.now()
        print("start_ng_samp:  ", time)
        train_loader.dataset.ng_sample()

        print("end_ng_samp:  ", time)

        epoch_loss = 0

        # -----------------------------------------------------------------------------------
        self.behavior_loss_list = [0] * len(self.behaviors)
        self.behavior_loss_list1 = [None] * len(self.behaviors)
        self.L2_loss_list = [None] * len(self.behaviors)

        self.user_id_list = [None] * len(self.behaviors)
        self.item_id_pos_list = [None] * len(self.behaviors)
        self.item_id_neg_list = [None] * len(self.behaviors)

        self.meta_start_index = 0
        self.meta_end_index = self.meta_start_index + args.meta_batch
        # ----------------------------------------------------------------------------------

        cnt = 0
        for user, item_i, item_j in tqdm(train_loader):

            user = user.long().cuda()
            self.user_step_index = user
            # ---round one---------------------------------------------------------------------------------------------

            meta_behavior_loss_list = [None] * len(self.behaviors)
            meta_user_index_list = [None] * len(self.behaviors)  # ---
            Cascading=0
            intent_behavior_loss_list = [None] * len(self.behaviors)
            intent_user_index_list = [None] * len(self.behaviors)  # ---
            L2_loss_list= [None] * len(self.behaviors)  # ---
            meta_user_embed, meta_item_embed, meta_user_embeds, meta_item_embeds= self.model()

            for index in range(len(self.behaviors)):
                not_zero_index = np.where(item_i[index].cpu().numpy() != -1)[0]

                self.user_id_list[index] = user[not_zero_index].long().cuda()
                meta_user_index_list[index] = self.user_id_list[index]
                self.item_id_pos_list[index] = item_i[index][not_zero_index].long().cuda()
                self.item_id_neg_list[index] = item_j[index][not_zero_index].long().cuda()

                meta_userEmbed = meta_user_embed[self.user_id_list[index]]
                meta_posEmbed = meta_item_embed[self.item_id_pos_list[index]]
                meta_negEmbed = meta_item_embed[self.item_id_neg_list[index]]

                # u_intent_em = u_targets[self.user_id_list[index]]
                # i_intentp_em = i_targets[self.item_id_pos_list[index]]
                # i_intentn_em = i_targets[self.item_id_neg_list[index]]

                meta_pred_i, meta_pred_j = 0, 0
                meta_pred_i, meta_pred_j = self.innerProduct(meta_userEmbed, meta_posEmbed, meta_negEmbed)
                # intent_pred_i, intent_pred_j = 0, 0
                # intent_pred_i, intent_pred_j = self.innerProduct(u_intent_em , i_intentp_em, i_intentn_em)

                meta_behavior_loss_list[index] = - ((meta_pred_i.view(-1) - meta_pred_j.view(-1))+1e-5).sigmoid().log()


            """sparse_tensor = self.behavior_mats[3]["A"]
            # 指定需要提取的行数
            rows = self.user_step_index.tolist()
            # 提取稀疏矩阵的指定行
            num_rows = len(rows)
            row_indices_list = []
            row_values_list = []
            for i, row in enumerate(rows):
                row_indices = sparse_tensor._indices()[0] == row
                row_values = sparse_tensor._values()[row_indices]
                row_indices = sparse_tensor._indices()[:, row_indices]
                row_indices[0, :] = i  # 重新编号每一行
                row_indices_list.append(row_indices)
                row_values_list.append(row_values)
            sparse_indices = torch.cat(row_indices_list, dim=1)
            sparse_values = torch.cat(row_values_list, dim=0)
            sparse_tensor_subset = torch.sparse.FloatTensor(sparse_indices, sparse_values, size=(num_rows, 31228))
            # 输出结果
            # print(sparse_tensor_subset.to_dense())
            oo = sparse_tensor_subset.transpose(0, 1)
            juzhen = torch.sparse.mm(sparse_tensor_subset, oo)
            juzhen = juzhen.to_dense()
            #values, indics, = torch.topk(juzhen.fill_diagonal_(0), 1)
            values, indics, = torch.topk(juzhen, 2)
            for i in range(indics.shape[0]):
                if values[i, 1]  <= 3:
                    indics[i, 1] = i"""
            #print(self.user_step_index)
            sparse_tensor = self.behavior_mats[3]["A"]
            # 指定需要提取的行数
            rows = self.user_step_index.tolist()
            # 提取稀疏矩阵的指定行
            num_rows = len(rows)
            row_indices_list = []
            row_values_list = []
            for i, row in enumerate(rows):
                row_indices = sparse_tensor._indices()[0] == row
                row_values = sparse_tensor._values()[row_indices]
                row_indices = sparse_tensor._indices()[:, row_indices]
                row_indices[0, :] = i  # 重新编号每一行
                row_indices_list.append(row_indices)
                row_values_list.append(row_values)
            sparse_indices = torch.cat(row_indices_list, dim=1)
            sparse_values = torch.cat(row_values_list, dim=0)
            sparse_tensor_subset = torch.sparse.FloatTensor(sparse_indices, sparse_values, size=(num_rows, 50000))
            # 输出结果
            # print(sparse_tensor_subset.to_dense())
            #oo = sparse_tensor_subset.transpose(0, 1)
            #juzhen = torch.sparse.mm(sparse_tensor_subset, oo)
            juzhen = sparse_tensor_subset.to_dense()
            juzhen = torch.tensor(cosine_similarity(juzhen.cpu()))
            #values, indics, = torch.topk(juzhen.fill_diagonal_(0), 1)
            
            #juzhen=sparse_tensor_subset.to_dense()
            #juzhen = torch.tensor(cosine_similarity(juzhen.cpu()))
            # 输出结果
            # print(sparse_tensor_subset.to_dense())
            #oo = sparse_tensor_subset.transpose(0, 1)
            #juzhen = torch.sparse.mm(sparse_tensor_subset, oo)
            #juzhen = juzhen.to_dense()
            #values, indics, = torch.topk(juzhen.fill_diagonal_(0), 1)
            values, indics, = torch.topk(juzhen, 2)
            #print(values)
            numbers=[]
            for i in range(indics.shape[0]):
                if values[i, 1]  <= 0.8:
                    indics[i, 1] = i
                else :
                    numbers.append(i)
            sparse_tensor = self.behavior_mats[0]["A"]
            # 指定需要提取的行数
            rows = self.user_step_index.tolist()
            # 提取稀疏矩阵的指定行
            num_rows = len(rows)
            row_indices_list = []
            row_values_list = []
            for i, row in enumerate(rows):
                row_indices = sparse_tensor._indices()[0] == row
                row_values = sparse_tensor._values()[row_indices]
                row_indices = sparse_tensor._indices()[:, row_indices]
                row_indices[0, :] = i  # 重新编号每一行
                row_indices_list.append(row_indices)
                row_values_list.append(row_values)
            sparse_indices = torch.cat(row_indices_list, dim=1)
            sparse_values = torch.cat(row_values_list, dim=0)
            sparse_tensor_subset = torch.sparse.FloatTensor(sparse_indices, sparse_values, size=(num_rows, 50000))
            # 输出结果
            # print(sparse_tensor_subset.to_dense())
            #oo = sparse_tensor_subset.transpose(0, 1)
            #juzhen = torch.sparse.mm(sparse_tensor_subset, oo)
            juzhen = sparse_tensor_subset.to_dense()
            juzhen = torch.tensor(cosine_similarity(juzhen.cpu()))
            #values, indics, = torch.topk(juzhen.fill_diagonal_(0), 1)
            
            #juzhen=sparse_tensor_subset.to_dense()
            #juzhen = torch.tensor(cosine_similarity(juzhen.cpu()))
            # 输出结果
            # print(sparse_tensor_subset.to_dense())
            #oo = sparse_tensor_subset.transpose(0, 1)
            #juzhen = torch.sparse.mm(sparse_tensor_subset, oo)
            #juzhen = juzhen.to_dense()
            #values, indics, = torch.topk(juzhen.fill_diagonal_(0), 1)
            values1, indics1, = torch.topk(juzhen, 2)
            #print(values)
            numbers=[]
            for i in range(indics1.shape[0]):
                if values1[i, 1]  <= 0.8:
                    indics1[i, 1] = i
                else :
                    numbers.append(i)
            d = torch.cat((indics, indics1), dim=1)
            #print(d)
            result = []


            for column in d:

                counter = Counter(column)
                most_common = counter.most_common()
                tuples_list = most_common[:2]
                output_list = [tup[0] for tup in tuples_list]
                result.append(output_list)
            e=torch.tensor(result)

         
            meta_infoNCELoss_list, SSL_user_step_index = self.SSL1(meta_user_embeds, self.user_step_index,e)
            
            for i in range(len(self.behaviors)):
                meta_infoNCELoss_list[i] = (meta_infoNCELoss_list[i]).sum()
                #self.behavior_loss_list[i]=(self.behavior_loss_list[i]).sum()
                
                #   meta_infoNCELoss_list1[i] = (meta_infoNCELoss_list1[i]).sum()
                meta_behavior_loss_list[i] = (meta_behavior_loss_list[i]).sum()

            #        intent_behavior_loss_list[i] = (intent_behavior_loss_list[i]).sum()
            meta_bprloss = sum(meta_behavior_loss_list) / len(meta_behavior_loss_list)
            #self.behavior_loss_list[-1]=(self.behavior_loss_list[-1]).sum()
            #c_bprloss = sum(self.behavior_loss_list) / len(self.behavior_loss_list)
            
            meta_infoNCELoss = sum(meta_infoNCELoss_list) / len(meta_infoNCELoss_list)
            # meta_infoNCELoss1 = sum(meta_infoNCELoss_list1) / len(meta_infoNCELoss_list1)
            meta_regLoss = (t.norm(meta_userEmbed) ** 2 + t.norm(meta_posEmbed) ** 2 + t.norm(meta_negEmbed) ** 2)
            #c_regLoss = (t.norm(c_uemb) ** 2 + t.norm(c_piemb) ** 2 + t.norm(c_niemb) ** 2)
            # intent_regLoss = (t.norm(u_intent_em) ** 2 + t.norm(i_intentp_em) ** 2 + t.norm(i_intentn_em) ** 2)
            # intent_regLoss = (t.norm(intentu) ** 2 + t.norm(intenti) ** 2)
            meta_model_loss = (meta_bprloss+args.reg * meta_regLoss + args.beta * meta_infoNCELoss) / args.batch
            epoch_loss = epoch_loss + meta_model_loss.item()
            #meta_opt.zero_grad(set_to_none=True)
            self.opt.zero_grad(set_to_none=True)
            meta_model_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()
            user_embed, item_embed, user_embeds, item_embeds = self.model()
        return epoch_loss, user_embed, item_embed, user_embeds, item_embeds

    def dcg_at_k(self, r, k, method=1):
        """Score is discounted cumulative gain (dcg)
        Relevance is positive real values.  Can use binary
        as the previous methods.
        Returns:
            Discounted cumulative gain
        """
        r = np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            else:
                raise ValueError('method must be 0 or 1.')
        return 0.

    def ndcg_at_k(self, r, k, method=1):
        """Score is normalized discounted cumulative gain (ndcg)
        Relevance is positive real values.  Can use binary
        as the previous methods.
        Returns:
            Normalized discounted cumulative gain
        """
        dcg_max = self.dcg_at_k(sorted(r, reverse=True), k, method)
        if not dcg_max:
            return 0.
        return self.dcg_at_k(r, k, method) / dcg_max

    def recall_at_k(self, r, k, all_pos_num):
        r = np.asfarray(r)[:k]
        return np.sum(r) / all_pos_num

    def hit_at_k(self, r, k):
        r = np.array(r)[:k]
        if np.sum(r) > 0:
            return 1.
        else:
            return 0.

    def precision_at_k(self, r, k):
        """Score is precision @ k
        Relevance is binary (nonzero is relevant).
        Returns:
            Precision @ k
        Raises:
            ValueError: len(r) must be >= k
        """
        assert k >= 1
        r = np.asarray(r)[:k]
        return np.mean(r)

    def get_performance(self, user_pos_test, r, auc, Ks):
        precision, recall, ndcg, hit_ratio = [], [], [], []

        for K in Ks:
            precision.append(self.precision_at_k(r, K))
            recall.append(self.recall_at_k(r, K, len(user_pos_test)))
            ndcg.append(self.ndcg_at_k(r, K))
            hit_ratio.append(self.hit_at_k(r, K))

        return {'recall': np.array(recall), 'precision': np.array(precision),
                'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}

    def auc(self, ground_truth, prediction):
        try:
            res = roc_auc_score(y_true=ground_truth, y_score=prediction)
        except Exception:
            res = 0.
        return res

    def get_auc(self, item_score, user_pos_test):
        item_score = sorted(item_score.items(), key=lambda kv: kv[1])
        item_score.reverse()
        item_sort = [x[0] for x in item_score]
        posterior = [x[1] for x in item_score]

        r = []
        for i in item_sort:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = self.auc(ground_truth=r, prediction=posterior)
        return auc

    def ranklist_by_sorted(self, user_pos_test, test_items, rating, Ks):
        item_score = {}
        for i in test_items:
            item_score[i] = rating[i]

        K_max = max(Ks)
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

        r = []
        for i in K_max_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = self.get_auc(item_score, user_pos_test)
        return r, auc

    def ranklist_by_heapq(self, user_pos_test, test_items, rating, Ks):
        item_score = {}
        for i in test_items:
            item_score[i] = rating[i]

        K_max = max(Ks)
        K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)  # 取出候选item中排名前K的item id

        r = []
        for i in K_max_item_score:
            if i in user_pos_test:
                r.append(1)
            else:
                r.append(0)
        auc = 0.
        return r, auc

    def test_one_user(self, x):
        # user u's ratings for user u
        rating = x[0]  # [1, N]
        # uid
        u = x[1]  # [1, ]
        # user u's items in the training set
        try:
            training_items = self.trnDicts[-1]
        except Exception:
            training_items = []
        # user u's items in the test set
        user_pos_test = self.tstDicts

        all_items = set(range(self.item_num))

        test_items = list(all_items - set(training_items))

        r, auc = self.ranklist_by_heapq(user_pos_test, test_items, rating, [10, 50])

        return self.get_performance(user_pos_test, r, auc, [10, 50])

    def testEpoch(self, data_loader, save=False):

        epochHR, epochNDCG = [0] * 2
        with t.no_grad():
            user_embed, item_embed, user_embeds, item_embeds= self.model()
            user_embed1, item_embed1, user_embeds, item_embeds= self.model1()
        user_embed=user_embed
        item_embed=item_embed
        cnt = 0
        tot = 0
        for user, item_i in data_loader:
            user_compute, item_compute, user_item1, user_item100 = self.sampleTestBatch(user, item_i)
            userEmbed = user_embed[user_compute]  # [614400, 16], [147894, 16]
            itemEmbed = item_embed[item_compute]

            pred_i = t.sum(t.mul(userEmbed, itemEmbed), dim=1)

            hit, ndcg = self.calcRes(t.reshape(pred_i, [user.shape[0], 100]), user_item1, user_item100)
            epochHR = epochHR + hit
            epochNDCG = epochNDCG + ndcg  #
            cnt += 1
            tot += user.shape[0]

        result_HR = epochHR / tot
        result_NDCG = epochNDCG / tot
        print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}")

        return result_HR, result_NDCG

    def calcRes(self, pred_i, user_item1, user_item100):  # [6144, 100] [6144] [6144, (ndarray:(100,))]

        hit = 0
        ndcg = 0

        for j in range(pred_i.shape[0]):

            _, shoot_index = t.topk(pred_i[j], args.shoot)
            shoot_index = shoot_index.cpu()
            shoot = user_item100[j][shoot_index]
            shoot = shoot.tolist()

            if type(shoot) != int and (user_item1[j] in shoot):
                hit += 1
                ndcg += np.reciprocal(np.log2(shoot.index(user_item1[j]) + 2))
            elif type(shoot) == int and (user_item1[j] == shoot):
                hit += 1
                ndcg += np.reciprocal(np.log2(0 + 2))

        return hit, ndcg  # int, float

    def sampleTestBatch(self, batch_user_id, batch_item_id):

        batch = len(batch_user_id)
        tmplen = (batch * 100)

        sub_trainMat = self.trainMat[batch_user_id].toarray()
        user_item1 = batch_item_id
        user_compute = [None] * tmplen
        item_compute = [None] * tmplen
        user_item100 = [None] * (batch)

        cur = 0
        for i in range(batch):
            pos_item = user_item1[i]
            negset = np.reshape(np.argwhere(sub_trainMat[i] == 0), [-1])
            pvec = self.labelP[negset]
            pvec = pvec / np.sum(pvec)

            random_neg_sam = np.random.permutation(negset)[:99]
            user_item100_one_user = np.concatenate((random_neg_sam, np.array([pos_item])))
            user_item100[i] = user_item100_one_user

            for j in range(100):
                user_compute[cur] = batch_user_id[i]
                item_compute[cur] = user_item100_one_user[j]
                cur += 1

        return user_compute, item_compute, user_item1, user_item100

    def setRandomSeed(self):
        np.random.seed(args.seed)
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)
        random.seed(args.seed)

    def getModelName(self):
        title = args.title
        ModelName = \
            args.point + \
            "_" + title + \
            "_" + args.dataset + \
            "_" + modelTime + \
            "_lr_" + str(args.lr) + \
            "_reg_" + str(args.reg) + \
            "_batch_size_" + str(args.batch) + \
            "_gnn_layer_" + str(args.gnn_layer)

        return ModelName

    def saveHistory(self):
        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        ModelName = self.modelName

        with open(r'./History/' + args.dataset + r'/' + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)

    def saveModel(self):
        ModelName = self.modelName

        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        savePath = r'./Model/' + args.dataset + r'/' + ModelName + r'.pth'
        params = {
            'epoch': self.curEpoch,
            # 'lr': self.lr,
            'model': self.model,
            # 'reg': self.reg,
            'history': history,
            'user_embed': self.user_embed,
            'user_embeds': self.user_embeds,
            'item_embed': self.item_embed,
            'item_embeds': self.item_embeds,
        }
        t.save(params, savePath)

    def loadModel(self, loadPath):
        ModelName = self.modelName
        # loadPath = r'./Model/' + args.dataset + r'/' + ModelName + r'.pth'
        loadPath = loadPath
        checkpoint = t.load(loadPath)
        self.model = checkpoint['model']

        self.curEpoch = checkpoint['epoch'] + 1
        # self.lr = checkpoint['lr']
        # self.args.reg = checkpoint['reg']
        history = checkpoint['history']
        self.train_loss = history['loss']
        self.his_hr = history['HR']
        self.his_ndcg = history['NDCG']
        # log("load model %s in epoch %d"%(modelPath, checkpoint['epoch']))




if __name__ == '__main__':
    print(args)
    my_model = Model()
    my_model.run()
    # my_model.test()

