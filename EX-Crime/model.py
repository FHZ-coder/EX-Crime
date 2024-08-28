import torch
import torch.nn as nn
from Params import args
import torch.nn.functional as F
from torch_cluster import random_walk
import numpy as np
import torch_sparse
# Local
# Local Spatial cnn
class spa_cnn_local(nn.Module):
    def __init__(self, input_dim, output_dim, ):
        super(spa_cnn_local, self).__init__()
        self.spaConv1 = nn.Conv3d(input_dim, output_dim, kernel_size=[args.kernelSize, args.kernelSize, args.cateNum], stride=1, padding=[int((args.kernelSize-1)/2), int((args.kernelSize-1)/2), 0])
        self.spaConv2 = nn.Conv3d(input_dim, output_dim, kernel_size=[args.kernelSize, args.kernelSize, args.cateNum], stride=1, padding=[int((args.kernelSize-1)/2), int((args.kernelSize-1)/2), 0])
        self.spaConv3 = nn.Conv3d(input_dim, output_dim, kernel_size=[args.kernelSize, args.kernelSize, args.cateNum], stride=1, padding=[int((args.kernelSize-1)/2), int((args.kernelSize-1)/2), 0])
        self.spaConv4 = nn.Conv3d(input_dim, output_dim, kernel_size=[args.kernelSize, args.kernelSize, args.cateNum], stride=1, padding=[int((args.kernelSize-1)/2), int((args.kernelSize-1)/2), 0])
        self.drop = nn.Dropout(args.dropRateL)
        self.act_lr = nn.LeakyReLU()

    def forward(self, embeds):
        cate_1 = self.drop(self.spaConv1(embeds))
        cate_2 = self.drop(self.spaConv2(embeds))
        cate_3 = self.drop(self.spaConv3(embeds))
        cate_4 = self.drop(self.spaConv4(embeds))
        spa_cate = torch.cat([cate_1, cate_2, cate_3, cate_4], dim=-1)
        return self.act_lr(spa_cate + embeds)

# Local Temporal cnn
class tem_cnn_local(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(tem_cnn_local, self).__init__()
        self.temConv1 = nn.Conv3d(input_dim, output_dim, stride=[1, 1, 1], kernel_size=[1, args.kernelSize, args.cateNum], padding=[0, int((args.kernelSize-1)/2), 0])
        self.temConv2 = nn.Conv3d(input_dim, output_dim, stride=[1, 1, 1], kernel_size=[1, args.kernelSize, args.cateNum], padding=[0, int((args.kernelSize-1)/2), 0])
        self.temConv3 = nn.Conv3d(input_dim, output_dim, stride=[1, 1, 1], kernel_size=[1, args.kernelSize, args.cateNum], padding=[0, int((args.kernelSize-1)/2), 0])
        self.temConv4 = nn.Conv3d(input_dim, output_dim, stride=[1, 1, 1], kernel_size=[1, args.kernelSize, args.cateNum], padding=[0, int((args.kernelSize-1)/2), 0])
        self.act_lr = nn.LeakyReLU()
        self.drop = nn.Dropout(args.dropRateL)

    def forward(self, embeds):
        cate_1 = self.drop(self.temConv1(embeds))
        cate_2 = self.drop(self.temConv2(embeds))
        cate_3 = self.drop(self.temConv3(embeds))
        cate_4 = self.drop(self.temConv4(embeds))
        tem_cate = torch.cat([cate_1, cate_2, cate_3, cate_4], dim=-1)
        return self.act_lr(tem_cate + embeds)


class GCNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = torch_sparse.spmm(adj.indices(), new_t, args.areNum,  args.areNum, x)
        return x

class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.layer1 = GCNLayer(input_dim, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, hidden_dim)
        self.layer3 = GCNLayer(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = self.layer2(x, adj)
        x = self.layer3(x, adj)
        return x

class SequentialMultipleInputs(nn.Sequential):
    def forward(self, x, adj):
        for module in self:
            x = module(x, adj)
        return x
    
class Normalgraph_IB(nn.Module):
    def __init__(self,embeddingsize,input_dim, hidden_dim, output_dim, adj_matrix, device,areaNum): 
        super(Normalgraph_IB,self).__init__()
        self.s_mask_learner = nn.ModuleList([nn.Sequential(nn.Linear(embeddingsize, embeddingsize), nn.ReLU(), nn.Linear(embeddingsize, 1)) for i in range(args.L)])
        self.t_mask_learner = nn.ModuleList([nn.Sequential(nn.Linear(2 * embeddingsize, embeddingsize), nn.ReLU(), nn.Linear(embeddingsize, 1)) for i in range(args.L)])
        self.adj_matrix=adj_matrix
        self.areaNum=areaNum
        self.create_sparse_adjaceny()
        self.device=device

    def create_sparse_adjaceny(self):
        nonzero_count = np.count_nonzero(self.adj_matrix)
        rows, cols = np.nonzero(self.adj_matrix)
        indices_list = [rows.tolist(), cols.tolist()]
        values = [1.0] * nonzero_count
        self.crime_matrix = torch.sparse_coo_tensor(indices_list, values, (self.areaNum,self.areaNum))
        degree = torch.sparse.sum(self.crime_matrix, dim=1).to_dense()
        degree = torch.pow(degree, -0.5)
        degree[torch.isinf(degree)] = 0 
        D_inverse = torch.diag(degree, diagonal=0).to_sparse()
        self.crime_matrix_normal = torch.sparse.mm(torch.sparse.mm(D_inverse, self.crime_matrix), D_inverse).coalesce()

        joint_indices =self.crime_matrix_normal.indices()
        self.row = joint_indices[0]
        self.col = joint_indices[1]
        start = torch.arange(self.areaNum)
        walk = random_walk(self.row, self.col, start, walk_length=args.walk_length)

        self.rw_adj = torch.zeros((self.areaNum, self.areaNum))
        self.rw_adj = torch.scatter(self.rw_adj, 1, walk, 1).to_sparse()
        degree = torch.sparse.sum(self.rw_adj, dim=1).to_dense()
        degree = torch.pow(degree, -1)
        degree[torch.isinf(degree)] = 0 
        D_inverse = torch.diag(degree, diagonal=0).to_sparse()
        self.rw_adj = torch.sparse.mm(D_inverse, self.rw_adj).to(args.device)   
        self.crime_matrix_normal= self.crime_matrix_normal.to(args.device)  


    def forward(self,crime_embedding):
        cur_embedding=crime_embedding
        all_embeddings = [crime_embedding]  #areas time crimetype 
        t_mask_list = []
        s_mask_list = []
        for i in range(args.L):
            cur_embedding = torch.mm(self.crime_matrix_normal, cur_embedding)
            all_embeddings.append(cur_embedding)
            t_cat_embedding = torch.cat([crime_embedding[self.row], crime_embedding[self.col]], dim=-1)
            t_mask = self.t_mask_learner[i](t_cat_embedding)
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(t_mask.size()) + (1 - bias)
            t_gate_inputs = torch.log(eps) - torch.log(1 - eps)
            t_gate_inputs = t_gate_inputs.to(args.device)
            t_gate_inputs = (t_gate_inputs + t_mask) / args.choosing_tmp
            t_mask = torch.sigmoid(t_gate_inputs).squeeze(1)
            t_mask_list.append(t_mask)

            s_mask = self.s_mask_learner[i](cur_embedding)
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(s_mask.size()) + (1 - bias)
            s_gate_inputs = torch.log(eps) - torch.log(1 - eps)
            s_gate_inputs = s_gate_inputs.to(args.device)
            s_gate_inputs = (s_gate_inputs + s_mask) / args.choosing_tmp
            s_mask = torch.sigmoid(s_gate_inputs)
            s_mask_list.append(s_mask)

        all_embeddings = torch.stack(all_embeddings, dim=0)
        all_embeddings = torch.mean(all_embeddings, dim=0, keepdim=False)
        
        cur_embedding_t=crime_embedding
        all_embeddings_t = [cur_embedding_t]

        t_reg = 0
        for i in range(args.L):
            t_mask = t_mask_list[i]
            new_t = torch.mul(self.crime_matrix_normal.values(), t_mask)  ##
            cur_embedding_t = torch_sparse.spmm(self.crime_matrix_normal.indices(), new_t, self.areaNum, self.areaNum, cur_embedding_t)
            

            all_embeddings_t.append(cur_embedding_t)

        all_embeddings_t = torch.stack(all_embeddings_t, dim=0)
        all_embeddings_t = torch.mean(all_embeddings_t, dim=0, keepdim=False)

        t_reg = t_reg / args.L
        cur_embedding_s=crime_embedding
        all_embeddings_s = [cur_embedding_s]


        s_reg = 0 

        for i in range(args.L):
            s_mask = s_mask_list[i]
            mean_pooling_embedding = torch.mm(self.rw_adj, cur_embedding_s)
            cur_embedding_s = torch.mul(s_mask, cur_embedding_s) + torch.mul((1-s_mask), mean_pooling_embedding)
            cur_embedding_s = torch.mm(self.crime_matrix_normal, cur_embedding_s)
            all_embeddings_s.append(cur_embedding_s)
            s_reg += s_mask.sum()/args.areaNum

        all_embeddings_s = torch.stack(all_embeddings_s, dim=0)
        all_embeddings_s = torch.mean(all_embeddings_s, dim=0, keepdim=False)
        
        s_reg = s_reg / args.L
        return all_embeddings_t, all_embeddings_s, t_reg, s_reg,t_mask

   

class IB_CDiff(nn.Module):
    def __init__(self,adj_matrix,device):
        super(IB_CDiff, self).__init__()
        self.dimConv_in = nn.Conv3d(1, args.latdim, kernel_size=1, padding=0, bias=True)
        self.dimConv_local = nn.Conv2d(args.latdim, 1, kernel_size=1, padding=0, bias=True)

        self.spa_cnn_local1 = spa_cnn_local(args.latdim, args.latdim)
        self.spa_cnn_local2 = spa_cnn_local(args.latdim, args.latdim)
        self.tem_cnn_local1 = tem_cnn_local(args.latdim, args.latdim)
        self.tem_cnn_local2 = tem_cnn_local(args.latdim, args.latdim)
        self.adj_matrix=adj_matrix
        self.device=device
        self.GIB=Normalgraph_IB(embeddingsize=args.channels,input_dim=args.channels, hidden_dim=args.channels, 
                       output_dim=args.channels,adj_matrix=self.adj_matrix,device=self.device,areaNum=args.areaNum)

    def forward(self, embeds_true):
        embeds_in_global = self.dimConv_in(embeds_true.unsqueeze(1))
        embeds_in_local = embeds_in_global.permute(0, 3, 1, 2, 4).contiguous().view(-1, args.latdim, args.row, args.col, 4)
        spa_local1 = self.spa_cnn_local1(embeds_in_local)
        spa_local2 = self.spa_cnn_local2(spa_local1)
        spa_local2 = spa_local2.view(-1, args.temporalRange, args.latdim, args.areaNum, args.cateNum).permute(0, 2, 3, 1, 4)
        tem_local1 = self.tem_cnn_local1(spa_local2)
        tem_local2 = self.tem_cnn_local2(tem_local1)
        eb_local = tem_local2.mean(3)
        out_local = self.dimConv_local(eb_local).squeeze(1)
        gib_input=out_local.view(args.areaNum,-1)
        all_embeddings_t, all_embeddings_s, t_reg, s_reg, t_mask=self.GIB(gib_input)  #gib_input
        return out_local,all_embeddings_t, all_embeddings_s, t_reg, s_reg, t_mask            

  