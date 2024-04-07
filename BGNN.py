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
    def __init__(self, userNum, itemNum, behavior, behavior_mats,R0, R1,R2,R3,R4, all_h_list0, all_h_list1, all_h_list2,all_h_list3,
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
        self.R4 = R4
        self.all_h_list0 = all_h_list0
        self.all_h_list1 = all_h_list1
        self.all_h_list2 = all_h_list2
        self.all_h_list3 = all_h_list3
        self.all_t_list0 = all_t_list0
        self.all_t_list1 = all_t_list1
        self.all_t_list2 = all_t_list2
        self.all_t_list3 = all_t_list3

        self.gcn = GCN(self.userNum, self.itemNum, self.behavior, self.behavior_mats, self.R0,self.R1,self.R2,self.R3,self.R4,self.all_h_list0,self.all_h_list1,self.all_h_list2,self.all_h_list3,self.all_t_list0,self.all_t_list1,self.all_t_list2,self.all_t_list3)


    def forward(self):

        user_embedding, item_embedding, user_embeddings, item_embeddings = self.gcn()
        return user_embedding, item_embedding, user_embeddings, item_embeddings


class GCN(nn.Module):
    def __init__(self, userNum, itemNum, behavior, behavior_mats,R0, R1,R2,R3,R4, all_h_list0, all_h_list1, all_h_list2,all_h_list3,
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
        self.R4 = self._convert_sp_mat_to_sp_tensor(R4).cuda()
        self.adj=[]
        self.adj.append(self.R0)
        self.adj.append(self.R1)
        self.adj.append(self.R2)
        self.adj.append(self.R3)
        self.all_h_list0 = all_h_list0
        self.all_h_list1 = all_h_list1
        self.all_h_list2 = all_h_list2
        self.all_h_list3 = all_h_list3

        self.all_t_list0 = all_t_list0
        self.all_t_list1 = all_t_list1
        self.all_t_list2 = all_t_list2
        self.all_t_list3 = all_t_list3

        self.user_embedding, self.item_embedding = self.init_embedding()

        self.gating_weightu1 = nn.Parameter(
            torch.FloatTensor(args.hidden_dim, args.hidden_dim))
        nn.init.xavier_normal_(self.gating_weightu1.data)

        self.gating_weightib1 = nn.Parameter(
            torch.FloatTensor(1, args.hidden_dim))
        nn.init.xavier_normal_(self.gating_weightib1.data)

        self.gating_weightu2 = nn.Parameter(
            torch.FloatTensor(args.hidden_dim, args.hidden_dim))
        nn.init.xavier_normal_(self.gating_weightu2.data)

        self.gating_weightib2 = nn.Parameter(
            torch.FloatTensor(1, args.hidden_dim))
        nn.init.xavier_normal_(self.gating_weightib2.data)

        self.gating_weightu3 = nn.Parameter(
            torch.FloatTensor(args.hidden_dim, args.hidden_dim))
        nn.init.xavier_normal_(self.gating_weightu3.data)

        self.gating_weightib3 = nn.Parameter(
            torch.FloatTensor(1, args.hidden_dim))
        nn.init.xavier_normal_(self.gating_weightib3.data)

        self.gating_weighti1 = nn.Parameter(
            torch.FloatTensor(args.hidden_dim, args.hidden_dim))
        nn.init.xavier_normal_(self.gating_weighti1.data)

        self.gating_weightbi1 = nn.Parameter(
            torch.FloatTensor(1, args.hidden_dim))
        nn.init.xavier_normal_(self.gating_weightbi1.data)

        self.gating_weighti2 = nn.Parameter(
            torch.FloatTensor(args.hidden_dim, args.hidden_dim))
        nn.init.xavier_normal_(self.gating_weighti2.data)

        self.gating_weightbi2 = nn.Parameter(
            torch.FloatTensor(1, args.hidden_dim))
        nn.init.xavier_normal_(self.gating_weightbi2.data)

        self.gating_weighti3 = nn.Parameter(
            torch.FloatTensor(args.hidden_dim, args.hidden_dim))
        nn.init.xavier_normal_(self.gating_weighti3.data)

        self.gating_weightbi3 = nn.Parameter(
            torch.FloatTensor(1, args.hidden_dim))
        nn.init.xavier_normal_(self.gating_weightbi3.data)

        self.i_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        self.u_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)
        self.ii_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        self.uu_w = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        init.xavier_uniform_(self.ii_w)
        init.xavier_uniform_(self.uu_w)
        
        self.beh0 = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        self.beh1 = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        self.beh2 = nn.Parameter(torch.Tensor(args.hidden_dim, args.hidden_dim))
        init.xavier_uniform_(self.beh0)
        init.xavier_uniform_(self.beh1)
        init.xavier_uniform_(self.beh2)
        
        self.encoder_u = nn.Sequential(
            nn.Linear(args.hidden_dim*2, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim))

        self.encoder_i = nn.Sequential(
            nn.Linear(args.hidden_dim*2, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim))   

        self.sigmoid = torch.nn.Sigmoid()
        self.act = torch.nn.PReLU()
        self.dropout = torch.nn.Dropout(args.drop_rate)
        self.gnn_layer = eval(args.gnn_layer)
        self.layers = nn.ModuleList()
        for i in range(0, len(self.gnn_layer)):
            # print("1")
            self.layers.append(GCNLayer(args.hidden_dim, args.hidden_dim, self.userNum, self.itemNum, self.behavior,
                                        self.behavior_mats))


    def init_embedding(self):
        user_embedding = torch.nn.Embedding(self.userNum, args.hidden_dim)
        item_embedding = torch.nn.Embedding(self.itemNum, args.hidden_dim)

        nn.init.xavier_uniform_(user_embedding.weight)
        nn.init.xavier_uniform_(item_embedding.weight)

        return user_embedding, item_embedding

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        shape = coo.shape
        return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape))

    def _adaptive_mask(self, head_embeddings, tail_embeddings, all_h_list, all_t_list,R):
        vals=torch.tensor(R.tocoo().data)
        edgeNum = vals.size()
        head_embeddings = torch.nn.functional.normalize(head_embeddings)
        tail_embeddings = torch.nn.functional.normalize(tail_embeddings)
        #edge_alpha =self.mlp_0(head_embeddings * tail_embeddings)
        edge_alpha = torch.sum(head_embeddings * tail_embeddings, dim=1)
        mask = ((edge_alpha + 1).floor()).type(torch.bool)
        #result_a = torch.masked_select(all_h_list,edge_alpha)
        #result_b = torch.masked_select(all_t_list, edge_alpha)

        result_a = all_h_list[mask]
        result_b = all_t_list[mask]

        newVals = vals[mask]
        newVals = newVals / (newVals.size()[0] / edgeNum[0])
        newIdxs = torch.stack([result_a, result_b], dim=0)
        #G_values = D_scores_inv[all_h_list] * edge_alpha


        #result_a = torch.masked_select(all_h_list,edge_alpha)
        #result_b = torch.masked_select(all_t_list, edge_alpha)

        #A_tensor = torch_sparse.SparseTensor(row=all_h_list, col=all_t_list, value=edge_alpha,
         #                                    sparse_sizes=self.R2.tocoo().shape).cuda()
        #print("CUDA:Ready")
        #D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)

        return torch.sparse.FloatTensor(newIdxs, newVals.cuda(), R.tocoo().shape).cuda()


    def denoise(self, origin_emb, target_emb):
        res_array = torch.unsqueeze(torch.sum(origin_emb * target_emb, dim=1), -1) * target_emb
        norm_num = torch.norm(target_emb, dim=1) * torch.norm(target_emb, dim=1) + 1e-12
        clear_emb = res_array / torch.unsqueeze(norm_num, -1)
        noise_emb = origin_emb - clear_emb
        if False:
            a = torch.where(torch.sum(origin_emb * target_emb, dim=1) >= 0, torch.tensor(1.0), torch.tensor(0.0))
            clear_emb *= torch.unsqueeze(a, -1)
        return clear_emb * 0.3, noise_emb * 0.3

    def forward(self, user_embedding_input=None, item_embedding_input=None):

        all_user_embeddings = []
        all_item_embeddings = []
        all_uu1_embeddings = []
        all_ii1_embeddings = []
        all_uu2_embeddings = []
        all_ii2_embeddings = []
        all_uu3_embeddings = []
        all_ii3_embeddings = []
        all_user_embeddingss = []
        all_item_embeddingss = []
        # all_uu1_embeddings = []
        # all_ii1_embeddings = []
        # print("222")
        all_aux0=[]
        all_aux1 = []
        all_aux2 = []
        all_aux3 = []

        user_embedding = self.user_embedding.weight
        item_embedding = self.item_embedding.weight
        user_embedding_c=user_embedding
        item_embedding_c=item_embedding
        allEmbed=torch.cat((user_embedding_c,item_embedding_c),dim=0)
        all_embeddings=[allEmbed]
        

        

        uu_embed0 = torch.multiply(user_embedding, self.sigmoid(
            torch.matmul(user_embedding, self.gating_weightu1) + self.gating_weightib1))
        ii_embed0 = torch.multiply(item_embedding, self.sigmoid(
            torch.matmul(item_embedding, self.gating_weighti1) + self.gating_weightbi1))
        uu_embed1 = torch.multiply(user_embedding, self.sigmoid(
            torch.matmul(user_embedding, self.gating_weightu2) + self.gating_weightib2))
        ii_embed1 = torch.multiply(item_embedding, self.sigmoid(
            torch.matmul(item_embedding, self.gating_weighti2) + self.gating_weightbi2))
        uu_embed2 = torch.multiply(user_embedding, self.sigmoid(
            torch.matmul(user_embedding, self.gating_weightu3) + self.gating_weightib3))
        ii_embed2 = torch.multiply(item_embedding, self.sigmoid(
            torch.matmul(item_embedding, self.gating_weighti3) + self.gating_weightbi3))
        all_user_embeddings.append(user_embedding)
        all_item_embeddings.append(item_embedding)
        all_uu1_embeddings.append(uu_embed0)
        all_ii1_embeddings.append(ii_embed0)
        all_uu2_embeddings.append(uu_embed1)
        all_ii2_embeddings.append(ii_embed1)
        all_uu3_embeddings.append(uu_embed2)
        all_ii3_embeddings.append(ii_embed2)

        for i, layer in enumerate(self.layers):
            user_embedding, item_embedding, user_embeddings, item_embeddings, uu_embed0, ii_embed0, uu_embed1, ii_embed1, uu_embed2, ii_embed2 = \
                layer(user_embedding, item_embedding, uu_embed0, ii_embed0, uu_embed1, ii_embed1, uu_embed2,
                      ii_embed2)
            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)
            all_uu1_embeddings.append(uu_embed0)
            all_ii1_embeddings.append(ii_embed0)
            all_uu2_embeddings.append(uu_embed1)
            all_ii2_embeddings.append(ii_embed1)
            all_uu3_embeddings.append(uu_embed2)
            all_ii3_embeddings.append(ii_embed2)
            all_user_embeddingss.append(user_embeddings)
            all_item_embeddingss.append(item_embeddings)
        user_embedding = torch.stack(all_user_embeddings, dim=1)
        item_embedding = torch.stack(all_item_embeddings, dim=1)

        uu_embed0 = torch.stack(all_uu1_embeddings, dim=1)
        ii_embed0 = torch.stack(all_ii1_embeddings, dim=1)
        uu_embed1 = torch.stack(all_uu2_embeddings, dim=1)
        ii_embed1 = torch.stack(all_ii2_embeddings, dim=1)
        uu_embed2 = torch.stack(all_uu3_embeddings, dim=1)
        ii_embed2 = torch.stack(all_ii3_embeddings, dim=1)

        user_embedding = torch.mean(user_embedding, dim=1)
        item_embedding = torch.mean(item_embedding, dim=1)
        uu_embed0 = torch.mean(uu_embed0, dim=1)
        ii_embed0 = torch.mean(ii_embed0, dim=1)
        uu_embed1 = torch.mean(uu_embed1, dim=1)
        ii_embed1 = torch.mean(ii_embed1, dim=1)

        uu_embed2 = torch.mean(uu_embed2, dim=1)
        ii_embed2 = torch.mean(ii_embed2, dim=1)
        
        

        
        

        user_embedding55=user_embedding
        item_embedding55=item_embedding       

        user_embedding = user_embedding + torch.cat(
            (uu_embed0.unsqueeze(dim=0), uu_embed1.unsqueeze(dim=0), uu_embed2.unsqueeze(dim=0)), dim=0).mean(dim=0)

        item_embedding = item_embedding + torch.cat(
            (ii_embed0.unsqueeze(dim=0),ii_embed1.unsqueeze(dim=0), ii_embed2.unsqueeze(dim=0)), dim=0).mean(dim=0)
        


        user_embedding = F.normalize(user_embedding)
        item_embedding = F.normalize(item_embedding)
        user_embeddings = torch.stack(all_user_embeddingss, dim=0).mean(dim=0)
        item_embeddings = torch.stack(all_item_embeddingss, dim=0).mean(dim=0)

        user_embedding = torch.matmul(user_embedding, self.uu_w)
        item_embedding = torch.matmul(item_embedding, self.ii_w)
        






        return user_embedding, item_embedding, user_embeddings, item_embeddings
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, userNum, itemNum, behavior, behavior_mats):
        super(GCNLayer, self).__init__()

        self.behavior = behavior
        self.behavior_mats = behavior_mats

        self.userNum = userNum
        self.itemNum = itemNum
        self.act1 = torch.nn.PReLU()
        self.act = torch.nn.Sigmoid()
        self.i_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.u_w = nn.Parameter(torch.Tensor(in_dim, out_dim))
        init.xavier_uniform_(self.i_w)
        init.xavier_uniform_(self.u_w)
        self.dropout = torch.nn.Dropout(args.drop_rate)
        
    def forward(self, user_embedding, item_embedding, uu_embed0, ii_embed0, uu_embed1, ii_embed1, uu_embed2,
                ii_embed2):

        user_embedding_list = [None] * len(self.behavior)
        item_embedding_list = [None] * len(self.behavior)

        for i in self.behavior:
            if i == "buy":
                user_embedding_list[3] = torch.spmm(self.behavior_mats[3]['A'], item_embedding)
                item_embedding_list[3] = torch.spmm(self.behavior_mats[3]['AT'], user_embedding)
            if i == "cart":
                user_embedding_list[2] = torch.spmm(self.behavior_mats[2]['A'], ii_embed2)
                item_embedding_list[2] = torch.spmm(self.behavior_mats[2]['AT'], uu_embed2)
            if i == "fav":
                user_embedding_list[1] = torch.spmm(self.behavior_mats[1]['A'], ii_embed1)
                item_embedding_list[1] = torch.spmm(self.behavior_mats[1]['AT'], uu_embed1)
            if i == "click":
                user_embedding_list[0] = torch.spmm(self.behavior_mats[0]['A'], ii_embed0)
                item_embedding_list[0] = torch.spmm(self.behavior_mats[0]['AT'], uu_embed0)

        user_embeddings = torch.stack(user_embedding_list, dim=0)
        item_embeddings = torch.stack(item_embedding_list, dim=0)





        user_embedding = user_embedding_list[3]
        item_embedding = item_embedding_list[3]
        uu_embed1 = user_embedding_list[1]
        ii_embed1 = item_embedding_list[1]
        # ['pv', 'fav', 'cart', 'buy']
        uu_embed0 = user_embedding_list[0]
        ii_embed0 = item_embedding_list[0]
        uu_embed2 = user_embedding_list[2]
        ii_embed2 = item_embedding_list[2]


        user_embedding =self.act(torch.matmul(torch.mean(user_embeddings, dim=0), self.u_w))
        item_embedding =self.act(torch.matmul(torch.mean(item_embeddings, dim=0), self.i_w))
  
        user_embeddings = F.normalize(user_embeddings)
        item_embeddings = F.normalize(item_embeddings)

        return user_embedding, item_embedding, user_embeddings, item_embeddings, uu_embed0, ii_embed0, uu_embed1, ii_embed1, uu_embed2, ii_embed2



