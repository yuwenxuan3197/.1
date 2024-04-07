import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from Params import args



def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class myModel(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats,R0, R1,R2,R3, all_h_list0, all_h_list1, all_h_list2,all_h_list3,
                 all_t_list0, all_t_list1, all_t_list2, all_t_list3):
        super(myModel, self).__init__()

        self.userNum = userNum
        self.itemNum = itemNum
        self.behavior = behavior
        self.behavior_mats = behavior_mats
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.all_h_list0 = all_h_list0
        self.all_h_list1 = all_h_list1
        self.all_h_list2 = all_h_list2
        self.all_h_list3 = all_h_list3
        self.all_t_list0 = all_t_list0
        self.all_t_list1 = all_t_list1
        self.all_t_list2 = all_t_list2
        self.all_t_list3 = all_t_list3
        self.gcn = GCN(self.userNum, self.itemNum, self.behavior, self.behavior_mats, self.R0,self.R1,self.R2,self.R3,self.all_h_list0,self.all_h_list1,self.all_h_list2,self.all_h_list3,self.all_t_list0,self.all_t_list1,self.all_t_list2,self.all_t_list3)

    def forward(self):

        embedding_total_u, embedding_total_i,embedding_ulist,embedding_ilist= self.gcn()
        return embedding_total_u, embedding_total_i,embedding_ulist,embedding_ilist




class GCN(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats,R0, R1,R2,R3, all_h_list0, all_h_list1, all_h_list2,all_h_list3,
                 all_t_list0, all_t_list1, all_t_list2, all_t_list3):
        super(GCN, self).__init__()
        self.userNum = userNum
        self.itemNum = itemNum
        self.hidden_dim = args.hidden_dim  # 64

        self.behavior = behavior
        self.behavior_mats = behavior_mats
        self.R0 = self._convert_sp_mat_to_sp_tensor(R0).cuda()
        self.R1 = self._convert_sp_mat_to_sp_tensor(R1).cuda()
        self.R2 = self._convert_sp_mat_to_sp_tensor(R2).cuda()
        self.R3 = self._convert_sp_mat_to_sp_tensor(R3).cuda()

        self.all_h_list0 = all_h_list0
        self.all_h_list1 = all_h_list1
        self.all_h_list2 = all_h_list2
        self.all_h_list3 = all_h_list3

        self.all_t_list0 = all_t_list0
        self.all_t_list1 = all_t_list1
        self.all_t_list2 = all_t_list2
        self.all_t_list3 = all_t_list3
        self.all_weights = {}
        self.all_weights['user_embeddings'] = nn.Parameter(torch.FloatTensor(self.userNum, self.hidden_dim).cuda())
        self.all_weights['item_embeddings'] = nn.Parameter(torch.FloatTensor(self.itemNum, self.hidden_dim).cuda())
        self.all_weights['W_u'] = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim).cuda())
        self.all_weights['W_i'] = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim).cuda())
        self.all_weights['WW_u'] = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim).cuda())
        self.all_weights['WW_i'] = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim).cuda())
        nn.init.xavier_uniform_(self.all_weights['user_embeddings'])
        nn.init.xavier_uniform_(self.all_weights['item_embeddings'])
        nn.init.xavier_uniform_(self.all_weights['W_u'])
        nn.init.xavier_uniform_(self.all_weights['W_i'])
        nn.init.xavier_uniform_(self.all_weights['WW_u'])
        nn.init.xavier_uniform_(self.all_weights['WW_i'])
        self.all_weights = nn.ParameterDict(self.all_weights)
        self.act = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(args.drop_rate)
        self.gnn_layer = eval(args.gnn_layer)
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        shape = coo.shape
        return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape))
    def forward(self, user_embedding_input=None, item_embedding_input=None):

        user_embedding = self.all_weights["user_embeddings"]
        item_embedding = self.all_weights["item_embeddings"]
        embedding_step_now=torch.cat((user_embedding,item_embedding),dim=0)
        embeddings_=embedding_step_now
        # pv fav cart buy
        for i in range(0,len(self.gnn_layer)):
            embeddings_ = torch.matmul(self.R0,embeddings_)
            embedding_step_now+=embeddings_
        embedding_step_now/=len(self.gnn_layer)+1
        # pv
        embedding_step_pv=torch.matmul(embedding_step_now, self.all_weights['W_u'])
        embeddings_pv = embedding_step_pv

        for i in range(0,len(self.gnn_layer)):
            embeddings_pv = torch.matmul(self.R1,embeddings_pv)
            embedding_step_pv+=embeddings_pv
        embedding_step_pv /= len(self.gnn_layer) + 1
        # fav
        embedding_step_fav=torch.matmul(embedding_step_pv, self.all_weights['W_i'])
        embeddings_fav = embedding_step_fav

        for i in range(0,len(self.gnn_layer)):
            embeddings_fav = torch.matmul(self.R2,embeddings_fav)
            embedding_step_fav+=embeddings_fav
        embedding_step_fav /= len(self.gnn_layer) + 1

        # cart
        embedding_step_cart=torch.matmul(embedding_step_fav, self.all_weights['WW_u'])
        embeddings_cart = embedding_step_cart
        for i in range(0,len(self.gnn_layer)):
            embeddings_cart = torch.matmul(self.R3,embeddings_cart)
            embedding_step_cart+=embeddings_cart
        embedding_step_cart /= len(self.gnn_layer) + 1


        embedding_total=embedding_step_now+embedding_step_pv+embedding_step_fav+embedding_step_cart
        embedding_total_u, embedding_total_i = torch.split(embedding_total, [self.userNum, self.itemNum], 0)

        embedding_ulist=[]
        embedding_ilist = []
        embedding_pv_u, embedding_pv_i = torch.split(embedding_step_now, [self.userNum, self.itemNum], 0)

        embedding_fav_u, embedding_fav_i = torch.split(embedding_step_pv, [self.userNum, self.itemNum], 0)
        embedding_cart_u, embedding_cart_i = torch.split(embedding_step_fav, [self.userNum, self.itemNum], 0)
        embedding_buy_u, embedding_buy_i = torch.split(embedding_step_cart, [self.userNum, self.itemNum], 0)
        embedding_ulist.append(embedding_pv_u)
        embedding_ulist.append(embedding_fav_u)
        embedding_ulist.append(embedding_cart_u)
        embedding_ulist.append(embedding_buy_u)

        embedding_ilist.append(embedding_pv_i)
        embedding_ilist.append(embedding_fav_i)
        embedding_ilist.append(embedding_cart_i)
        embedding_ilist.append(embedding_buy_i)







        return embedding_total_u, embedding_total_i,embedding_ulist,embedding_ilist