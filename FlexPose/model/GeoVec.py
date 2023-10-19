import torch
from einops import rearrange, repeat
import numpy as np


EPS = 1e-8    # Warning: used carefully for avoiding nan


def make_embed(input_channel, output_channel):
    return torch.nn.Sequential(
        torch.nn.Linear(input_channel, output_channel),
        torch.nn.LeakyReLU())


class VNL(torch.nn.Module): # VectorNonLinear
    def __init__(self, in_hidden, out_hidden, leaky_relu=False):
        super(VNL, self).__init__()
        self.leaky_relu = leaky_relu
        self.lin_vec = torch.nn.Linear(in_hidden, out_hidden, bias=False)
        if leaky_relu==True:
            assert in_hidden==out_hidden

    def forward(self, vec):
        # vec: vector features of shape [..., N_samples, N_feat, 3]
        vec_out = self.lin_vec(vec.transpose(-2, -1)).transpose(-2, -1)
        if self.leaky_relu:
            vec_dot = (vec * vec_out).sum(dim=-1, keepdim=True)
            mask = (vec_dot >= 0).to(vec.dtype)
            vec_out_norm_sq = (vec_out * vec_out).sum(dim=-1, keepdim=True)
            # >0: vec
            # <0: vec - 0.99*[vector of A map (projection) to B]
            # P.S. vec_out * vec_dot / vec_out_norm_sq
            #      = B * A.dot(B) / B.dot(B) (A:vec, B:vec_out)
            #      = B * |A|*|B|*cos(theta) / |B|*|B| (theta: angle for A and B)
            #      = B/|B| * |A|*cos(theta)
            #      = [normed_B] * [length for vec_A map (projection) to vec_B]
            #      = [vector of A map (projection) to B]
            vec_out = 0.01 * vec + \
                      0.99 * (mask * vec + (1 - mask) * (vec - vec_out * vec_dot / (vec_out_norm_sq + EPS)))
        return vec_out


class GVL(torch.nn.Module): # Equ tested
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector):
        super(GVL, self).__init__()
        dim_hid = max(in_vector, out_vector)
        self.lin_vec_1 = VNL(in_vector, dim_hid, leaky_relu=False)
        self.lin_vec_2 = VNL(dim_hid, out_vector, leaky_relu=False)
        self.sca_to_vec_gate = torch.nn.Linear(out_scalar, out_vector)
        self.lin_sca = torch.nn.Linear(in_scalar + dim_hid, out_scalar, bias=True)

    def forward(self, sca_vec):
        # vec: [..., N_samples, N_feat, 3]
        sca, vec = sca_vec
        vec_1 = self.lin_vec_1(vec)
        vec_1_norm = torch.norm(vec_1, p=2, dim=-1)
        sca_vec_cat = torch.cat([vec_1_norm, sca], dim=-1)

        sca_out = self.lin_sca(sca_vec_cat)
        vec_2 = self.lin_vec_2(vec_1)

        g = self.sca_to_vec_gate(sca_out).unsqueeze(dim=-1).sigmoid()
        vec_out = g * vec_2
        return (sca_out, vec_out)


class GVP(torch.nn.Module): # Equ tested
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector):
        super(GVP, self).__init__()
        self.lin = GVL(in_scalar, in_vector, out_scalar, out_vector)
        self.act_sca = torch.nn.LeakyReLU()
        self.act_vec = VNL(out_vector, out_vector, leaky_relu=True)

    def forward(self, sca_vec):
        sca, vec = self.lin(sca_vec)
        vec = self.act_vec(vec)
        sca = self.act_sca(sca)
        return (sca, vec)


class VecExpansion(torch.nn.Module): # Equ tested
    def __init__(self, in_vector, out_vector, norm=False):
        super(VecExpansion, self).__init__()
        # [..., N_samples, in_vector, 3] To [..., N_samples, out_vector, 3]
        self.norm = norm
        self.lin = torch.nn.Linear(in_vector, out_vector, bias=False)

    def forward(self, vec):
        if self.norm:
            non_zero_mask = (vec != 0).to(vec.dtype)
            vec = vec / (vec.norm(p=2, dim=-1, keepdim=True) + EPS) * non_zero_mask
        vec_expansion = self.lin(vec.transpose(-2, -1)).transpose(-2, -1)
        return vec_expansion


class VecLayerNorm(torch.nn.Module): # Equ tested
    def __init__(self):
        super(VecLayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(1))
        self.beta = torch.nn.Parameter(torch.zeros(1))

    def forward(self, vec):
        non_zero_mask = (vec != 0).to(vec.dtype)
        normed_vec = vec / (
                ((vec**2).sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True) + EPS) / vec.size(-2)
        )**0.5 * non_zero_mask
        normed_vec = normed_vec * self.gamma + self.beta
        return normed_vec


class DistGMM(torch.nn.Module):
    def __init__(self, start=0., stop=5., num_gaussians=50):
        super(DistGMM, self).__init__()
        self.stop = stop
        self.offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (self.offset[1] - self.offset[0]).item()**2
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, dist):
        encode_dist = dist
        encode_dist = encode_dist.clamp_max(self.stop)
        encode_dist = encode_dist.unsqueeze(dim=-1) - self.offset.to(dist.device)
        encode_dist = torch.cat([dist.unsqueeze(dim=-1) * self.scale,
                                 torch.exp(self.coeff * torch.pow(encode_dist, 2))],
                                dim=-1)
        return encode_dist


class GVDropout(torch.nn.Module):
    def __init__(self, dropout=0., sync=False):
        super(GVDropout, self).__init__()
        self.dropout = dropout
        self.sync = sync
        self.dropout_layer = torch.nn.Dropout(dropout)
        
    def forward(self, sca_vec):
        if self.dropout==0:
            return sca_vec
        
        sca, vec = sca_vec
        if self.sync:
            mask = torch.ones(sca.size(0), sca.size(1))
            mask = self.dropout_layer(mask)
            sca = sca * mask.unsqueeze(-1)
            vec = vec * rearrange(mask, 'b n -> b n () ()')
        else:
            sca = self.dropout_layer(sca)
            vec = self.dropout_layer(vec)
        return (sca, vec)


class GVMessage(torch.nn.Module):
    def __init__(self,
                 x_sca_hidden,
                 x_vec_hidden,
                 edge_sca_hidden,
                 edge_vec_hidden,
                 n_head,
                 dropout,
                 add_coor=False,
                 update_edge=False
                 ):
        super(GVMessage, self).__init__()
        self.add_coor = add_coor
        self.update_edge = update_edge
        self.n_head = n_head

        sca_mix_hidden = x_sca_hidden
        vec_mix_hidden = x_vec_hidden
        assert sca_mix_hidden % n_head == 0 and vec_mix_hidden % n_head == 0
        self.sca_head_hidden = sca_mix_hidden // n_head
        self.vec_head_hidden = vec_mix_hidden // n_head
        self.sqrt_head_hidden_sca = np.sqrt(self.sca_head_hidden)
        self.sqrt_head_hidden_vec = np.sqrt(self.vec_head_hidden * 3)
        self.lin_x_qkv = GVL(x_sca_hidden, x_vec_hidden, sca_mix_hidden*3, vec_mix_hidden*3)

        if add_coor:
            self.lin_edge_k = GVL(edge_sca_hidden*2, edge_vec_hidden*2, sca_mix_hidden, vec_mix_hidden)
            self.new_dist_embed = DistGMM(start=0., stop=2., num_gaussians=edge_sca_hidden-1)
            self.new_vec_embed = VecExpansion(1, edge_vec_hidden)
        else:
            self.lin_edge_k = GVL(edge_sca_hidden, edge_vec_hidden, sca_mix_hidden, vec_mix_hidden)

        self.lin_x_out = GVP(sca_mix_hidden, vec_mix_hidden, x_sca_hidden, x_vec_hidden)
        if update_edge:
            self.lin_edge_out = GVP(sca_mix_hidden, vec_mix_hidden, edge_sca_hidden, edge_vec_hidden)

        self.dropout = torch.nn.Dropout(dropout) # GVDropout(dropout=dropout)

    def forward(self, x_sca_vec, edge_sca_vec, mask, coor=None):
        x_sca, x_vec = x_sca_vec
        edge_sca, edge_vec = edge_sca_vec

        x_sca_qkv, x_vec_qkv = self.lin_x_qkv((x_sca, x_vec))
        x_sca_q, x_sca_k, x_sca_v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.n_head), x_sca_qkv.chunk(3, dim=-1))
        x_vec_q, x_vec_k, x_vec_v = map(lambda t: rearrange(t, 'b n (h d) c -> b n h d c', h=self.n_head), x_vec_qkv.chunk(3, dim=-2))

        if self.add_coor:
            rel_vec = coor.unsqueeze(-2) - coor.unsqueeze(-3)
            new_dist = self.new_dist_embed(rel_vec.norm(p=2, dim=-1))
            new_vec = self.new_vec_embed(rel_vec.unsqueeze(dim=-2))
            edge_sca = torch.cat([edge_sca, new_dist], dim=-1)
            edge_vec = torch.cat([edge_vec, new_vec], dim=-2)

        e_sca_b, e_vec_b = self.lin_edge_k((edge_sca, edge_vec))
        e_sca_b = rearrange(e_sca_b, 'b i j (h d) -> b i j h d', h=self.n_head)
        e_vec_b = rearrange(e_vec_b, 'b i j (h d) c -> b i j h d c', h=self.n_head)

        raw_att_sca = x_sca_q.unsqueeze(2) * x_sca_k.unsqueeze(1) * e_sca_b
        att_sca = raw_att_sca.sum(dim=-1) / self.sqrt_head_hidden_sca
        att_sca.masked_fill_(~repeat(mask, 'b j -> b i j ()', i=mask.size(1)), -torch.finfo(att_sca.dtype).max)
        att_sca = att_sca.softmax(dim=-2)
        att_sca = self.dropout(att_sca)
        raw_att_vec = x_vec_q.unsqueeze(2) * x_vec_k.unsqueeze(1) * e_vec_b
        att_vec = torch.einsum('b i j h d c -> b i j h', raw_att_vec) / self.sqrt_head_hidden_vec
        att_vec.masked_fill_(~repeat(mask, 'b j -> b i j ()', i=mask.size(1)), -torch.finfo(att_vec.dtype).max)
        att_vec = att_vec.softmax(dim=-2)
        att_vec = self.dropout(att_vec)

        x_sca_out = torch.einsum('b i j h, b j h d -> b i h d', att_sca, x_sca_v)
        x_vec_out = torch.einsum('b i j h, b j h d c -> b i h d c', att_vec, x_vec_v)

        x_sca_out = rearrange(x_sca_out, 'b i h d -> b i (h d)')
        x_vec_out = rearrange(x_vec_out, 'b i h d c -> b i (h d) c')
        x_sca_out, x_vec_out = self.lin_x_out((x_sca_out, x_vec_out))

        if self.update_edge:
            edge_sca_out = rearrange(raw_att_sca, 'b i j h d -> b i j (h d)', h=self.n_head)
            edge_vec_out = rearrange(raw_att_vec, 'b i j h d c -> b i j (h d) c', h=self.n_head)
            edge_sca_out, edge_vec_out = self.lin_edge_out((edge_sca_out, edge_vec_out))
            return (x_sca_out, x_vec_out), (edge_sca_out, edge_vec_out)
        else:
            return (x_sca_out, x_vec_out), None


class GVFeedForward(torch.nn.Module):
    def __init__(self,
                 sca_hidden,
                 vec_hidden,
                 dropout,
                 multi=1,
                 ):
        super(GVFeedForward, self).__init__()
        self.ff_1 = GVP(sca_hidden, vec_hidden, sca_hidden * multi, vec_hidden * multi)
        self.ff_2 = GVL(sca_hidden * multi, vec_hidden * multi, sca_hidden, vec_hidden)
        self.dropout = GVDropout(dropout=dropout)

    def forward(self, sca_vec):
        sca_vec = self.ff_1(sca_vec)
        sca_vec = self.dropout(sca_vec)
        sca_vec = self.ff_2(sca_vec)
        return sca_vec


class GVGateResidue(torch.nn.Module):
    def __init__(self, sca_hidden, vec_hidden, full_gate=True):
        super(GVGateResidue, self).__init__()
        self.full_gate = full_gate
        if full_gate:
            self.gate = GVL(sca_hidden * 3, vec_hidden * 3, sca_hidden, vec_hidden)
        else:
            self.gate = torch.nn.Linear(sca_hidden * 3 + vec_hidden * 3, 2)

    def forward(self, sca_vec, res_sca_vec):
        sca, vec = sca_vec
        res_sca, res_vec = res_sca_vec

        if self.full_gate:
            g_sca, g_vec = self.gate((torch.cat([sca, res_sca, sca - res_sca], dim=-1),
                                      torch.cat([vec, res_vec, vec - res_vec], dim=-2)))
            g_sca, g_vec = g_sca.sigmoid(), g_vec.norm(p=2, dim=-1, keepdim=True).sigmoid()
        else:
            sca_vec_cat = torch.cat([sca, vec.norm(p=2, dim=-1)], dim=-1)
            res_sca_vec_cat = torch.cat([res_sca, res_vec.norm(p=2, dim=-1)], dim=-1)
            g = self.gate(torch.cat([sca_vec_cat, res_sca_vec_cat, sca_vec_cat - res_sca_vec_cat], dim=-1))
            g_sca, g_vec = g.sigmoid().chunk(2, dim=-1)
            g_vec = g_vec.unsqueeze(dim=-1)
        out_sca = sca * g_sca + res_sca * (1 - g_sca)
        out_vec = vec * g_vec + res_vec * (1 - g_vec)
        return (out_sca, out_vec)


class GVNorm(torch.nn.Module):
    def __init__(self, sca_hidden):
        super(GVNorm, self).__init__()
        self.sca_norm = torch.nn.LayerNorm(sca_hidden)
        self.vec_norm = VecLayerNorm()

    def forward(self, sca_vec):
        sca, vec = sca_vec
        sca = self.sca_norm(sca)
        vec = self.vec_norm(vec)
        return (sca, vec)


class GVGateNormFeedForward(torch.nn.Module):
    def __init__(self, sca_hidden, vec_hidden, dropout, full_gate=True):
        super(GVGateNormFeedForward, self).__init__()
        self.ff = GVFeedForward(sca_hidden, vec_hidden, dropout)
        self.gate = GVGateResidue(sca_hidden, vec_hidden, full_gate)
        self.norm = GVNorm(sca_hidden)

    def forward(self, sca_vec):
        res_sca_vec = sca_vec
        sca_vec = self.ff(sca_vec)
        sca_vec = self.gate(sca_vec, res_sca_vec)
        sca_vec = self.norm(sca_vec)
        return sca_vec


class GateNormMessage(torch.nn.Module):
    def __init__(self,
                 x_sca_hidden,
                 x_vec_hidden,
                 edge_sca_hidden,
                 edge_vec_hidden,
                 n_head,
                 dropout,
                 full_gate=True,
                 add_coor=False,
                 update_edge=False
                 ):
        super(GateNormMessage, self).__init__()
        self.update_edge = update_edge
        self.message = GVMessage(x_sca_hidden,
                                 x_vec_hidden,
                                 edge_sca_hidden,
                                 edge_vec_hidden,
                                 n_head,
                                 dropout,
                                 add_coor=add_coor,
                                 update_edge=update_edge)
        self.node_gate = GVGateResidue(x_sca_hidden, x_vec_hidden, full_gate)
        self.node_norm = GVNorm(x_sca_hidden)
        if update_edge:
            self.edge_gate = GVGateResidue(edge_sca_hidden, edge_vec_hidden, full_gate)
            self.edge_norm = GVNorm(edge_sca_hidden)

    def forward(self, x_sca_vec, edge_sca_vec, mask, coor=None):
        res_x_sca_vec = x_sca_vec
        res_edge_sca_vec = edge_sca_vec

        x_sca_vec, edge_sca_vec = self.message(x_sca_vec, edge_sca_vec, mask, coor)

        x_sca_vec = self.node_gate(x_sca_vec, res_x_sca_vec)
        x_sca_vec = self.node_norm(x_sca_vec)

        if self.update_edge:
            edge_sca_vec = self.edge_gate(edge_sca_vec, res_edge_sca_vec)
            edge_sca_vec = self.edge_norm(edge_sca_vec)
        else:
            edge_sca_vec = res_edge_sca_vec
        return (x_sca_vec, edge_sca_vec)


class GVGraphTransformerBlock(torch.nn.Module):
    def __init__(self,
                 x_sca_hidden,
                 x_vec_hidden,
                 edge_sca_hidden,
                 edge_vec_hidden,
                 n_head,
                 dropout,
                 full_gate=True,
                 add_coor=False,
                 update_edge=False):
        super(GVGraphTransformerBlock, self).__init__()
        self.update_edge = update_edge
        self.message_update = GateNormMessage(x_sca_hidden, x_vec_hidden,
                                              edge_sca_hidden, edge_vec_hidden,
                                              n_head, dropout, full_gate=full_gate,
                                              add_coor=add_coor, update_edge=update_edge)
        self.node_ff = GVGateNormFeedForward(x_sca_hidden, x_vec_hidden, dropout, full_gate=full_gate)
        if update_edge:
            self.edge_ff = GVGateNormFeedForward(edge_sca_hidden, edge_vec_hidden, dropout, full_gate=full_gate)

    def forward(self, x_sca_vec, edge_sca_vec, mask, coor=None):
        x_sca_vec, edge_sca_vec = self.message_update(x_sca_vec, edge_sca_vec, mask, coor)
        x_sca_vec = self.node_ff(x_sca_vec)
        if self.update_edge:
            edge_sca_vec = self.edge_ff(edge_sca_vec)
        return (x_sca_vec, edge_sca_vec)


class CoorNorm(torch.nn.Module):
    def __init__(self):
        super(CoorNorm, self).__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, rel_coor):
        norm = rel_coor.norm(p=2, dim=-1, keepdim=True)
        norm = torch.where(norm == 0, norm + 1e+8, norm)  # for norm=0 (rel_coor with same atoms)
        normed_rel_coor = rel_coor / norm.clamp(min=1e-8)
        return normed_rel_coor * self.scale


class UpdateCoor(torch.nn.Module):
    def __init__(self,
                 x_sca_hidden,
                 x_vec_hidden,
                 edge_sca_hidden,
                 edge_vec_hidden,
                 n_head,
                 dropout,
                 only_update_coor=False,
                 update_coor_clamp=None,
                 use_raw_att=True):
        super(UpdateCoor, self).__init__()
        self.n_head = n_head
        self.only_update_coor = only_update_coor
        self.update_coor_clamp = update_coor_clamp
        self.use_raw_att = use_raw_att

        # for feat input
        sca_mix_hidden = x_sca_hidden
        vec_mix_hidden = x_vec_hidden
        assert sca_mix_hidden % n_head == 0 and vec_mix_hidden % n_head == 0
        self.sca_head_hidden = sca_mix_hidden // n_head
        self.vec_head_hidden = vec_mix_hidden // n_head
        self.sqrt_head_hidden_sca = np.sqrt(self.sca_head_hidden)
        self.sqrt_head_hidden_vec = np.sqrt(self.vec_head_hidden * 3)
        if only_update_coor:
            self.lin_x_qkv = GVL(x_sca_hidden, x_vec_hidden, sca_mix_hidden * 3, vec_mix_hidden * 3)
        else:
            self.lin_x_qkv = GVL(x_sca_hidden, x_vec_hidden, sca_mix_hidden * 4, vec_mix_hidden * 4)

        # fusion of spatial info
        dist_embed_dim = sca_mix_hidden
        vec_embed_dim = vec_mix_hidden
        self.dist_embed = DistGMM(start=0., stop=2., num_gaussians=dist_embed_dim - 1)
        self.vec_embed = VecExpansion(1, vec_embed_dim)
        self.lin_edge_b = GVL(edge_sca_hidden + dist_embed_dim, edge_vec_hidden + vec_embed_dim, sca_mix_hidden, vec_mix_hidden)

        # feat output
        if not only_update_coor:
            self.lin_x_out = GVP(sca_mix_hidden, vec_mix_hidden, x_sca_hidden, x_vec_hidden)
            self.lin_edge_out = GVP(sca_mix_hidden, vec_mix_hidden, edge_sca_hidden, edge_vec_hidden)

        # coor output
        self.lin_coor_out = torch.nn.Sequential(
            GVP(self.sca_head_hidden, self.vec_head_hidden, self.sca_head_hidden, self.vec_head_hidden),
            GVL(self.sca_head_hidden, self.vec_head_hidden, self.sca_head_hidden, 1))
        if use_raw_att:
            self.lin_att_coor_out = torch.nn.Sequential(torch.nn.Linear(self.sca_head_hidden + self.vec_head_hidden, self.sca_head_hidden + self.vec_head_hidden),
                                                        torch.nn.LeakyReLU(),
                                                        torch.nn.Linear(self.sca_head_hidden + self.vec_head_hidden, 1))
            self.coor_norm = CoorNorm()
            self.coor_head_combine = torch.nn.Parameter(torch.randn(n_head*2) * 1e-2)
        else:
            self.coor_head_combine = torch.nn.Parameter(torch.randn(n_head) * 1e-2)

        # dropout
        self.dropout = torch.nn.Dropout(dropout)  # GVDropout(dropout=dropout)

    def forward(self, x_sca_vec, edge_sca_vec, mask, coor, update_mask=None):
        x_sca, x_vec = x_sca_vec
        edge_sca, edge_vec = edge_sca_vec

        x_sca_qkv, x_vec_qkv = self.lin_x_qkv((x_sca, x_vec))
        if self.only_update_coor:
            x_sca_q, x_sca_k, x_sca_coor = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.n_head), x_sca_qkv.chunk(3, dim=-1))
            x_vec_q, x_vec_k, x_vec_coor = map(lambda t: rearrange(t, 'b n (h d) c -> b n h d c', h=self.n_head), x_vec_qkv.chunk(3, dim=-2))
        else:
            x_sca_q, x_sca_k, x_sca_v, x_sca_coor = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.n_head), x_sca_qkv.chunk(4, dim=-1))
            x_vec_q, x_vec_k, x_vec_v, x_vec_coor = map(lambda t: rearrange(t, 'b n (h d) c -> b n h d c', h=self.n_head), x_vec_qkv.chunk(4, dim=-2))

        rel_vec = coor.unsqueeze(-2) - coor.unsqueeze(-3)
        new_dist = self.dist_embed(rel_vec.norm(p=2, dim=-1))
        new_vec = self.vec_embed(rel_vec.unsqueeze(dim=-2))
        edge_sca = torch.cat([edge_sca, new_dist], dim=-1)
        edge_vec = torch.cat([edge_vec, new_vec], dim=-2)
        edge_sca_b, edge_vec_b = self.lin_edge_b((edge_sca, edge_vec))
        edge_sca_b = rearrange(edge_sca_b, 'b i j (h d) -> b i j h d', h=self.n_head)
        edge_vec_b = rearrange(edge_vec_b, 'b i j (h d) c -> b i j h d c', h=self.n_head)

        # raw attention
        raw_att_sca = x_sca_q.unsqueeze(2) * x_sca_k.unsqueeze(1) * edge_sca_b
        att_sca = torch.einsum('b i j h d -> b i j h', raw_att_sca) / self.sqrt_head_hidden_sca
        att_sca.masked_fill_(~rearrange(mask, 'b j -> b () j ()'), -torch.finfo(att_sca.dtype).max)
        att_sca = att_sca.softmax(dim=-2)
        att_sca = self.dropout(att_sca)
        raw_att_vec = x_vec_q.unsqueeze(2) * x_vec_k.unsqueeze(1) * edge_vec_b
        att_vec = torch.einsum('b i j h d c -> b i j h', raw_att_vec) / self.sqrt_head_hidden_vec
        att_vec.masked_fill_(~rearrange(mask, 'b j -> b () j ()'), -torch.finfo(att_vec.dtype).max)
        att_vec = att_vec.softmax(dim=-2)
        att_vec = self.dropout(att_vec)

        # coor
        x_sca_coor_out = torch.einsum('b i j h, b j h d -> b i h d', att_sca, x_sca_coor)
        x_vec_coor_out = torch.einsum('b i j h, b j h d c -> b i h d c', att_vec, x_vec_coor)
        # x_sca_coor_out = rearrange(x_sca_coor_out, 'b i h d -> b i (h d)')
        # x_vec_coor_out = rearrange(x_vec_coor_out, 'b i h d c -> b i (h d) c')
        _, x_vec_coor_out = self.lin_coor_out((x_sca_coor_out, x_vec_coor_out))
        x_vec_coor_out = x_vec_coor_out.squeeze(-2)

        if self.use_raw_att:
            rel_vec_norm = self.coor_norm(rel_vec)  # to avoid coordinate exploding, use CoorNorm
            att_coor_out = self.lin_att_coor_out(torch.cat([raw_att_sca, raw_att_vec.norm(p=2, dim=-1)], dim=-1)).squeeze(dim=-1)
            att_coor_out = att_coor_out * rearrange(mask, 'b j -> b () j ()')
            att_coor_out = torch.einsum('b i j h, b i j c -> b i h c', att_coor_out, rel_vec_norm)
            coor_out = torch.einsum('b i h c, h -> b i c', torch.cat([x_vec_coor_out, att_coor_out], dim=-2), self.coor_head_combine)
        else:
            coor_out = torch.einsum('b i h c, h -> b i c', x_vec_coor_out, self.coor_head_combine)

        if not isinstance(update_mask, type(None)):
            coor_out = coor_out * rearrange(update_mask, 'b n -> b n ()')
        if not isinstance(self.update_coor_clamp, type(None)):
            coor_out.clamp(min=-self.update_coor_clamp, max=self.update_coor_clamp)
        coor = coor + coor_out

        # output feat
        if not self.only_update_coor:
            x_sca_out = torch.einsum('b i j h, b j h d -> b i h d', att_sca, x_sca_v)
            x_vec_out = torch.einsum('b i j h, b j h d c -> b i h d c', att_vec, x_vec_v)

            x_sca_out = rearrange(x_sca_out, 'b i h d -> b i (h d)')
            x_vec_out = rearrange(x_vec_out, 'b i h d c -> b i (h d) c')
            x_sca_out, x_vec_out = self.lin_x_out((x_sca_out, x_vec_out))

            edge_sca_out = rearrange(raw_att_sca, 'b i j h d -> b i j (h d)', h=self.n_head)
            edge_vec_out = rearrange(raw_att_vec, 'b i j h d c -> b i j (h d) c', h=self.n_head)
            edge_sca_out, edge_vec_out = self.lin_edge_out((edge_sca_out, edge_vec_out))
        else:
            x_sca_out = None
            x_vec_out = None
            edge_sca_out = None
            edge_vec_out = None

        return (x_sca_out, x_vec_out), (edge_sca_out, edge_vec_out), coor


class GateNormUpdateCoor(torch.nn.Module):
    def __init__(self,
                 x_sca_hidden,
                 x_vec_hidden,
                 edge_sca_hidden,
                 edge_vec_hidden,
                 n_head,
                 dropout,
                 full_gate=True,
                 only_update_coor=False,
                 update_coor_clamp=None):
        super(GateNormUpdateCoor, self).__init__()
        self.only_update_coor = only_update_coor
        self.coor_update_layer = UpdateCoor(
            x_sca_hidden,
            x_vec_hidden,
            edge_sca_hidden,
            edge_vec_hidden,
            n_head,
            dropout,
            only_update_coor=only_update_coor,
            update_coor_clamp=update_coor_clamp)
        if not only_update_coor:
            self.x_gate = GVGateResidue(x_sca_hidden, x_vec_hidden, full_gate=full_gate)
            self.x_norm = GVNorm(x_sca_hidden)
            self.x_ff = GVGateNormFeedForward(x_sca_hidden, x_vec_hidden, dropout, full_gate=full_gate)
            self.edge_gate = GVGateResidue(edge_sca_hidden, edge_vec_hidden, full_gate=full_gate)
            self.edge_norm = GVNorm(edge_sca_hidden)
            self.edge_ff = GVGateNormFeedForward(edge_sca_hidden, edge_vec_hidden, dropout, full_gate=full_gate)

    def forward(self, x_sca_vec, edge_sca_vec, mask, coor, update_mask=None):
        res_x_sca_vec = x_sca_vec
        res_edge_sca_vec = edge_sca_vec

        x_sca_vec, edge_sca_vec, coor = self.coor_update_layer(x_sca_vec, edge_sca_vec, mask, coor, update_mask)
        if self.only_update_coor:
            return res_x_sca_vec, res_edge_sca_vec, coor
        else:
            x_sca_vec = self.x_gate(x_sca_vec, res_x_sca_vec)
            x_sca_vec = self.x_norm(x_sca_vec)
            x_sca_vec = self.x_ff(x_sca_vec)
            edge_sca_vec = self.edge_gate(edge_sca_vec, res_edge_sca_vec)
            edge_sca_vec = self.edge_norm(edge_sca_vec)
            edge_sca_vec = self.edge_ff(edge_sca_vec)
            return x_sca_vec, edge_sca_vec, coor





