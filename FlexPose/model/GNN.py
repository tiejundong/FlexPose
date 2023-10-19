import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np


class GateResidue(torch.nn.Module):
    def __init__(self, hidden, full_gate=True):
        super(GateResidue, self).__init__()
        if full_gate:
            self.gate = torch.nn.Linear(hidden*3, hidden)
        else:
            self.gate = torch.nn.Linear(hidden * 3, 1)

    def forward(self, x, res):
        g = self.gate(torch.cat((x, res, x-res), dim=-1)).sigmoid()
        return x*g + res*(1-g)


class FeedForward(torch.nn.Module):
    def __init__(self, hidden, dropout, multi=1):
        super(FeedForward, self).__init__()
        self.FF_1 = torch.nn.Linear(hidden, hidden*multi)
        self.act = torch.nn.LeakyReLU()
        self.dropout_rate = dropout
        self.dropout_layer = torch.nn.Dropout(self.dropout_rate)
        self.FF_2 = torch.nn.Linear(hidden*multi, hidden)

    def forward(self, x):
        x = self.FF_1(x)
        x = self.act(x)
        if self.dropout_rate > 0:
            x = self.dropout_layer(x)
        x = self.FF_2(x)
        return x


class GateNormFeedForward(torch.nn.Module):
    def __init__(self, hidden, dropout, full_gate=True):
        super(GateNormFeedForward, self).__init__()
        self.FF = FeedForward(hidden, dropout)
        self.gate = GateResidue(hidden, full_gate=full_gate)
        self.norm = torch.nn.LayerNorm(hidden)

    def forward(self, x):
        x_shortcut = x
        x = self.FF(x)
        x = self.gate(x, x_shortcut)
        x = self.norm(x)
        return x


class GAT(torch.nn.Module):
    def __init__(self,
                 node_hidden,
                 edge_hidden,
                 n_head,
                 head_hidden,
                 dropout):
        super(GAT, self).__init__()
        self.n_head = n_head
        self.head_hidden = head_hidden
        self.sqrt_head_hidden = np.sqrt(self.head_hidden)

        self.lin_qkv = torch.nn.Linear(node_hidden, n_head * head_hidden * 3)
        self.lin_edge = torch.nn.Linear(edge_hidden, n_head * head_hidden)
        self.lin_node_out = torch.nn.Linear(n_head * head_hidden, node_hidden)
        self.lin_edge_out = torch.nn.Linear(n_head * head_hidden, edge_hidden)

        self.dropout_rate = dropout
        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, x, edge_attr, mask):
        q, k, v = self.lin_qkv(x).chunk(3, dim=-1)
        e = self.lin_edge(edge_attr)
        v = rearrange(v, 'b j (h d) -> b j h d', h=self.n_head)
        e = rearrange(e, 'b i j (h d) -> b i j h d', h=self.n_head)

        q = rearrange(q, 'b i (h d) -> b i () h d', h=self.n_head)
        k = rearrange(k, 'b j (h d) -> b () j h d', h=self.n_head)
        raw_att = q * k * e

        att = raw_att.sum(dim=-1) / self.sqrt_head_hidden
        att.masked_fill_(~rearrange(mask, 'b j -> b () j ()'), -torch.finfo(att.dtype).max)
        att = att.softmax(dim=-2)
        if self.dropout_rate > 0:
            att = self.dropout_layer(att)
        node_out = torch.einsum('b i j h, b j h d -> b i h d', att, v)
        node_out = self.lin_node_out(rearrange(node_out, 'b i h d -> b i (h d)'))
        edge_out = self.lin_edge_out(rearrange(raw_att, 'b i j h d -> b i j (h d)'))

        return node_out, edge_out


class GraphTransformerBlock(torch.nn.Module):
    def __init__(self,
                 node_hidden,
                 edge_hidden,
                 n_head,
                 head_hidden,
                 dropout,
                 full_gate=True,
                 ):
        super(GraphTransformerBlock, self).__init__()
        self.att_layer = GAT(node_hidden,
                             edge_hidden,
                             n_head,
                             head_hidden,
                             dropout,)
        self.gate_node = GateResidue(node_hidden, full_gate=full_gate)
        self.gate_edge = GateResidue(edge_hidden, full_gate=full_gate)
        self.norm_node = torch.nn.LayerNorm(node_hidden)
        self.norm_edge = torch.nn.LayerNorm(edge_hidden)

    def forward(self, x, edge_attr, mask):
        x_shortcut = x
        edge_attr_shortcut = edge_attr

        # att
        x, edge_attr = self.att_layer(x, edge_attr, mask)
        # node & edge
        x = self.gate_node(x, x_shortcut)
        edge_attr = self.gate_edge(edge_attr, edge_attr_shortcut)
        x = self.norm_node(x)
        edge_attr = self.norm_edge(edge_attr)
        return x, edge_attr


class GraphTransformer(torch.nn.Module):
    def __init__(self,
                 n_block,
                 node_hidden,
                 edge_hidden,
                 n_head,
                 head_hidden,
                 dropout,
                 full_gate=True,
                 return_graph_pool=False):
        super(GraphTransformer, self).__init__()
        self.n_block = n_block
        self.return_graph_pool = return_graph_pool
        self.MP_layers = torch.nn.ModuleList([
            GraphTransformerBlock(
                node_hidden,
                edge_hidden,
                n_head,
                head_hidden,
                dropout,
                full_gate=full_gate
            ) for _ in range(self.n_block)])
        self.node_FF_layers = torch.nn.ModuleList([GateNormFeedForward(node_hidden, dropout, full_gate=full_gate) for _ in range(self.n_block)])
        self.edge_FF_layers = torch.nn.ModuleList([GateNormFeedForward(edge_hidden, dropout, full_gate=full_gate) for _ in range(self.n_block)])
        if return_graph_pool:
            self.lin_x_pool = torch.nn.Sequential(
                torch.nn.Linear(node_hidden, node_hidden),
                torch.nn.LeakyReLU(),
            )
            self.lin_edge_pool = torch.nn.Sequential(
                torch.nn.Linear(edge_hidden, edge_hidden),
                torch.nn.LeakyReLU()
            )

    def forward(self, x, edge_attr, x_mask, edge_mask):
        for i in range(self.n_block):
            x, edge_attr = self.MP_layers[i](x, edge_attr, x_mask)
            x = self.node_FF_layers[i](x)
            edge_attr = self.edge_FF_layers[i](edge_attr)
        if self.return_graph_pool:
            x_pool = (self.lin_x_pool(x) * x_mask.unsqueeze(-1)).sum(-2) / x_mask.sum(-1, keepdims=True)
            edge_pool = torch.einsum('b i j d -> b d', self.lin_edge_pool(edge_attr) * edge_mask.unsqueeze(-1)) / \
                        torch.einsum('b i j -> b', edge_mask).unsqueeze(-1)
            graph_pool = torch.cat([x_pool, edge_pool], dim=-1)
            return x, edge_attr, graph_pool
        else:
            return x, edge_attr



