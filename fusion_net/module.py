## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Written by Ce Zheng (cezheng@knights.ucf.edu)
# Modified by Qitao Zhao (qitaozhao@mail.sdu.edu.cn)
from functools import partial
from einops import rearrange
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import scipy.sparse as sp

import numpy as np
from timm.layers import DropPath

#### MLP Regression Head ####
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
#### End MLP Regression Head ####

#### PoseTransformer ####
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class PoseTransformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, out_chans=3, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, use_checkpoint=True):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        # out_dim = num_joints * 3     #### output dimension is num_joints * 3
        out_dim = out_chans * num_joints     #### output dimension is num_joints * 3

        ### spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        ####### A easy way to implement weighted mean
        self.weighted_mean = torch.nn.Conv1d(in_channels=num_frame, out_channels=num_frame, kernel_size=1)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )
        self.use_checkpoint = use_checkpoint

    def Spatial_forward_features(self, x):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def forward_features(self, x):
        b = x.shape[0]
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                # checkpointing requires the forward to be stateless wrt module
                x = cp.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        x = self.weighted_mean(x)
        return x


    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        # b, _, _, p = x.shape
        b, c, f, p = x.shape # changed
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Spatial_forward_features(x)
        x = self.forward_features(x)
        x = self.head(x)
        x = x.reshape(b, f, p, -1)

        return x
#### End PoseTransformer ####

#### DenseFC Regression Head ####
def linear_block(input_size, output_size, p_dropout):
    return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LayerNorm([output_size]),
            nn.PReLU(),
            nn.Dropout(p_dropout),
        )

class DenseFC(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 linear_size=512,
                 num_stage=3,
                 p_dropout=0.5,
                 ):
        super(DenseFC, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        self.input_size = in_channel
        self.output_size = out_channel

        self.dense_in = nn.Sequential(
            nn.Linear(self.input_size, self.linear_size),
            nn.LayerNorm(self.linear_size),
        )

        self.linear_stages = []
        for i in range(num_stage):
            self.linear_stages.append(linear_block(linear_size * (1 + i), linear_size, p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.dense_out = nn.Sequential(
            nn.Linear(self.linear_size * (self.num_stage + 1), self.output_size),
        )

    def forward(self, input):

        x = self.dense_in(input)
        for blk in self.linear_stages:
            y = blk(x)
            # Concatenate the input and output of each block on the channel dimension
            x = torch.cat((x, y), dim=-1)
        output = self.dense_out(x)

        return output
#### End DenseFC Regression Head ####

#### ResidualFC ####
class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()

        self.l_size = linear_size
        self.p_dropout = p_dropout

        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        self.dense_1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm2d(243)

        self.dense_2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm2d(243)

    def forward(self, x):
        y = self.dense_1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.dense_2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

class ResidualFC(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 linear_size=2048,
                 num_stage=3,  # original 2
                 p_dropout=0.5,
                 ):
        super(ResidualFC, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints + depth
        self.input_size = in_channel
        # 3d joints
        self.output_size = out_channel

        self.dense_in = nn.Sequential(
            nn.Linear(self.input_size, self.linear_size),
            nn.BatchNorm2d(243),
            nn.LeakyReLU(),
            nn.Dropout(self.p_dropout),
        )

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        self.dense_out = nn.Sequential(
            nn.Linear(self.linear_size, self.output_size),
        )

    def forward(self, input):

        output = self.dense_in(input)

        for i in range(self.num_stage):
            output = self.linear_stages[i](output)

        output = self.dense_out(output)

        return output

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in')
            # nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        else:
            pass
        return 0
#### End ResidualFC ####

#### DenseGCN ####
class ModulatedGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(ModulatedGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))  #torch.Size([2,2, 384])
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.zeros(size=(adj.size(0), out_features), dtype=torch.float))#17,384,取值在
        nn.init.xavier_uniform_(self.M.data, gain=1.414)

        self.adj = adj

        self.adj2 = nn.Parameter(torch.ones_like(adj))
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])  #input 256,17,2  -> 256,17,384
        h1 = torch.matmul(input, self.W[1])

        adj = self.adj.to(input.device) + self.adj2.to(input.device)
        adj = (adj.T + adj)/2
        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device) #17*17的I

        output = torch.matmul(adj * E, self.M*h0) + torch.matmul(adj * (1 - E), self.M*h1) #前者是专门针对自己的I，后者是针对M的
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1) #torch.Size([256, 17, 384])，全部都有的
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    

class DenseGCN(nn.Module):
    def __init__(self, adj, in_dim, out_dim, inter_dim, num_layer, drop_path=0., norm_layer=nn.LayerNorm):
        super(DenseGCN, self).__init__()
        self.num_layer = num_layer
        self.in_dim = in_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adj = adj
        self.norm_gcn1 = norm_layer(120)  # default 17
        self.gcn1 = ModulatedGraphConv(in_dim, inter_dim, self.adj)
        self.gelu = nn.GELU()

        self.gcn_layers = []
        for i in range(num_layer):
            self.gcn_layers.append(gcn_block(adj, (inter_dim * (1 + i)), inter_dim, self.drop_path, norm_layer))
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

        self.gcn_final = ModulatedGraphConv(inter_dim * (num_layer + 1), out_dim, self.adj)

    def forward(self, x_gcn):
        x_gcn = x_gcn.reshape(-1, 17, self.in_dim)
        x_gcn = self.drop_path(self.gelu(self.gcn1(self.norm_gcn1(x_gcn))))

        for layer in self.gcn_layers:
            y = layer(x_gcn)
            x_gcn = torch.cat((x_gcn, y), dim=2)

        x_gcn = self.gcn_final(x_gcn)
        return x_gcn.reshape(-1, 17 * 3)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_in')
            # nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        else:
            pass
        return 0


def gcn_block(adj, input_size, output_size, p_dropout, norm_layer):
    return nn.Sequential(
        norm_layer(input_size),
        ModulatedGraphConv(input_size, output_size, adj),
        nn.GELU(),
        p_dropout,
    )

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx) #+ sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)

    adj_mx = adj_mx * (1-torch.eye(adj_mx.shape[0])) + torch.eye(adj_mx.shape[0])
    return adj_mx

def adj_mx_from_skeleton():
    num_joints = 17
    parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12, 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27,
               28, 27, 30]
    # parents = (0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), parents)))
    # edges = [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6), (0, 7), (7, 17), (17, 8), (8, 9), (9, 10), (8, 14),
    #          (14, 15), (15, 16), (8, 11), (11, 12), (12, 13)]

    adj = adj_mx_from_edges(num_joints, edges, sparse=False)

    return adj
#### End DenseGCN ####