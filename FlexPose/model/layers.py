import torch
import torch.nn.functional as F
from torch_scatter import scatter_min, scatter_mean, scatter_add, scatter_softmax, scatter_max
from einops import rearrange, repeat, reduce

from model.GeoVec import *
from model.GNN import *
from model.MMFF import *
from model.min import *
from utils.common import *
from utils.hub import *


class PocketEncoder(torch.nn.Module):
    def __init__(self, args):
        super(PocketEncoder, self).__init__()
        self.p_x_sca_embed = make_embed(args.p_x_sca_indim + 1, args.p_x_sca_hidden)
        self.p_edge_sca_embed = make_embed(args.p_edge_sca_indim, args.p_edge_sca_hidden)
        self.p_x_vec_embed = VecExpansion(args.p_x_vec_indim, args.p_x_vec_hidden)
        self.p_edge_vec_embed = VecExpansion(args.p_edge_vec_indim, args.p_edge_vec_hidden)
        self.GVGraphTransformer_layers = torch.nn.ModuleList(
            [GVGraphTransformerBlock(
                args.p_x_sca_hidden,
                args.p_x_vec_hidden,
                args.p_edge_sca_hidden,
                args.p_edge_vec_hidden,
                args.n_head,
                args.dropout,
                full_gate=True,
                add_coor=False,
                update_edge=True
            ) for _ in range(args.p_block-1)] +
        [GVGraphTransformerBlock(
            args.p_x_sca_hidden,
            args.p_x_vec_hidden,
            args.p_edge_sca_hidden,
            args.p_edge_vec_hidden,
            args.n_head,
            args.dropout,
            full_gate=True,
            add_coor=False,
            update_edge=False
        )])

    def forward(self, complex_graph):
        complex_graph.p_x_sca = self.p_x_sca_embed(complex_graph.p_x_sca_init)
        complex_graph.p_edge_sca = self.p_edge_sca_embed(complex_graph.p_edge_sca_init)
        complex_graph.p_x_vec = self.p_x_vec_embed(complex_graph.p_x_vec_init)
        complex_graph.p_edge_vec = self.p_edge_vec_embed(complex_graph.p_edge_vec_init)

        complex_graph.p_x_sca_vec = (complex_graph.p_x_sca, complex_graph.p_x_vec)
        complex_graph.p_edge_sca_vec = (complex_graph.p_edge_sca, complex_graph.p_edge_vec)
        complex_graph.p_coor = complex_graph.p_coor_init  # useless in this module

        for i in range(len(self.GVGraphTransformer_layers)):
            complex_graph.p_x_sca_vec, complex_graph.p_edge_sca_vec = self.GVGraphTransformer_layers[i](
                complex_graph.p_x_sca_vec, complex_graph.p_edge_sca_vec, complex_graph.p_x_mask, coor=None)
        complex_graph.p_x_sca, complex_graph.p_x_vec = complex_graph.p_x_sca_vec
        complex_graph.p_edge_sca, complex_graph.p_edge_vec = complex_graph.p_edge_sca_vec

        return complex_graph


class LigandFeatEncoder(torch.nn.Module):
    def __init__(self, args):
        super(LigandFeatEncoder, self).__init__()
        self.l_x_sca_embed = make_embed(args.l_x_sca_indim + 1, args.l_x_sca_hidden)
        self.l_edge_sca_embed = make_embed(args.l_edge_sca_indim + 1, args.l_edge_sca_hidden)
        self.GTlayers = GraphTransformer(
            args.l_feat_block,
            args.l_x_sca_hidden,
            args.l_edge_sca_hidden,
            args.n_head,
            args.l_x_sca_hidden//args.n_head,
            args.dropout,
            full_gate=True,
            return_graph_pool=False
        )

    def forward(self, complex_graph):
        complex_graph.l_x_sca = self.l_x_sca_embed(complex_graph.l_x_sca_init)
        complex_graph.l_edge_sca = self.l_edge_sca_embed(complex_graph.l_edge_sca_init)

        complex_graph.l_x_sca, complex_graph.l_edge_sca = self.GTlayers(
            complex_graph.l_x_sca, complex_graph.l_edge_sca, complex_graph.l_x_mask, complex_graph.l_edge_mask)

        return complex_graph


class ComplexDecoder(torch.nn.Module):
    def __init__(self, args):
        super(ComplexDecoder, self).__init__()
        self.fix_prot = True if 'fix_prot' in vars(args).keys() and args.fix_prot else False
        self.GVGraphTransformer_layers = torch.nn.ModuleList(
            [GateNormUpdateCoor(
                args.c_x_sca_hidden,
                args.c_x_vec_hidden,
                args.c_edge_sca_hidden,
                args.c_edge_vec_hidden,
                args.n_head,
                args.dropout,
                full_gate=True,
                only_update_coor=False,
                update_coor_clamp=args.update_coor_clamp
            ) for _ in range(args.c_block)] +
            [GateNormUpdateCoor(
                args.c_x_sca_hidden,
                args.c_x_vec_hidden,
                args.c_edge_sca_hidden,
                args.c_edge_vec_hidden,
                args.n_head,
                args.dropout,
                full_gate=True,
                only_update_coor=True,
                update_coor_clamp=args.update_coor_clamp
            ) for _ in range(args.c_block_only_coor)])

    def forward(self, complex_graph):
        if self.fix_prot:
            # only for testing as baseline
            update_mask = complex_graph.l_coor_update_mask.float()
        else:
            update_mask = complex_graph.x_mask.float()
        coor_hidden = []
        for i in range(len(self.GVGraphTransformer_layers)):
            complex_graph.x_sca_vec, complex_graph.edge_sca_vec, complex_graph.coor = self.GVGraphTransformer_layers[i](
                complex_graph.x_sca_vec, complex_graph.edge_sca_vec, complex_graph.x_mask, complex_graph.coor,
                update_mask=update_mask)
            coor_hidden.append(complex_graph.coor)
        complex_graph.coor_hidden = torch.stack(coor_hidden, dim=-2)
        return complex_graph



class FlexPose(torch.nn.Module):
    def __init__(self, args=None, param_path=None):
        super(FlexPose, self).__init__()
        if args is not None:
            self.init_param(args)
        else:
            self.init_param_with_save(param_path)

    def forward(self, complex_graph, explicit_cycle=False, cycle_i=0, args=None, epoch=1e+5):
        # for pregen data
        if self.do_pregen_data:
            complex_graph = self.pretrain(complex_graph)
            return complex_graph

        # with explicit cycle
        # if explicit_cycle:
        #     if cycle_i == 0:
        #         complex_graph = self.pretrain(complex_graph)
        #         complex_graph = self.init_embed(complex_graph)
        #     complex_graph = self.run_cycle(complex_graph, cycle_i)
        #     tup_pred = self.pred_label(complex_graph)
        #     return tup_pred

        # first embed
        complex_graph = self.init_embed(complex_graph)

        # MC cycle
        complex_graph = self.run_cycling(complex_graph)

        # prediction
        tup_pred = self.pred_label(complex_graph)

        return tup_pred

    def run_cycling(self, complex_graph):
        # MC cycle
        if self.training:
            cycle_num = random.sample(range(1, self.n_cycle + 1), 1)[0]
            for cycle_i in range(cycle_num - 1):
                with torch.no_grad():
                    complex_graph = self.run_single_cycle(complex_graph, cycle_i)
                if self.use_min and cycle_i > 0:
                    complex_graph = self.energy_min(complex_graph)
            complex_graph = self.run_single_cycle(complex_graph, cycle_num - 1)
        else:
            for cycle_i in range(self.n_cycle):
                complex_graph = self.run_single_cycle(complex_graph, cycle_i)
                if self.use_min and cycle_i > 0 and cycle_i < self.n_cycle-1:
                    complex_graph = self.energy_min(complex_graph)
        return complex_graph

    def init_param(self, args):
        self.args = args
        self.n_cycle = args.n_cycle
        self.use_pretrain = args.use_pretrain
        self.do_pregen_data = args.do_pregen_data  # do pre-generation
        self.use_pregen_data = args.use_pregen_data  # use pre-generated data
        self.add_l_dismap = args.add_l_dismap
        self.coor_scale = args.coor_scale

        # pretrained
        if self.use_pretrain:
            self.p_encoder = PocketEncoder(args)
            self.l_feat_encoder = LigandFeatEncoder(args)
            self.load_encoder(args)

        # decoder
        self.c_decoder = ComplexDecoder(args)

        # E min
        self.use_min = args.MMFF_min
        if self.use_min:
            self.coor_min_object = CoorMin(args)

        # embedding
        # extra embedding for encoder (pretrain) and decoder
        if args.use_pretrain:
            # ligand embed
            self.l_extra_embed = True if args.l_x_sca_hidden != args.c_x_sca_hidden else False
            if self.l_extra_embed:
                self.l_x_sca_embed = make_embed(args.l_x_sca_hidden, args.c_x_sca_hidden)
            if self.add_l_dismap:
                self.l_edge_sca_embed = make_embed(args.l_edge_sca_hidden + 1, args.c_edge_sca_hidden)
            else:
                self.l_edge_sca_embed = make_embed(args.l_edge_sca_hidden, args.c_edge_sca_hidden)
            self.l_x_vec_embed = VecExpansion(args.l_x_vec_indim, args.c_x_vec_hidden)
            self.l_edge_vec_embed = VecExpansion(args.l_edge_vec_indim, args.c_edge_vec_hidden)

            # pocekt embed
            self.p_extra_embed = True if args.p_x_sca_hidden != args.c_x_sca_hidden else False
            self.p_x_sca_embed = make_embed(args.p_x_sca_hidden + 12, args.c_x_sca_hidden)  # explicit torsion
            if self.p_extra_embed:
                self.p_edge_sca_embed = make_embed(args.p_edge_sca_hidden, args.c_edge_sca_hidden)
                self.p_x_vec_embed = VNL(args.p_x_vec_hidden, args.c_x_vec_hidden, leaky_relu=False)
                self.p_edge_vec_embed = VNL(args.p_edge_vec_hidden, args.c_edge_vec_hidden, leaky_relu=False)
        else:
            # ligand embed
            self.l_x_sca_embed = make_embed(args.l_x_sca_indim + 1, args.c_x_sca_hidden)
            if self.add_l_dismap:
                self.l_edge_sca_embed = make_embed(args.l_edge_sca_indim + 1 + 1, args.c_edge_sca_hidden)
            else:
                self.l_edge_sca_embed = make_embed(args.l_edge_sca_indim + 1, args.c_edge_sca_hidden)
            self.l_x_vec_embed = VecExpansion(args.l_x_vec_indim, args.c_x_vec_hidden)
            self.l_edge_vec_embed = VecExpansion(args.l_edge_vec_indim, args.c_edge_vec_hidden)

            # pocekt embed
            self.p_x_sca_embed = make_embed(args.p_x_sca_indim + 12 + 1, args.c_x_sca_hidden)  # explicit torsion input
            self.p_edge_sca_embed = make_embed(args.p_edge_sca_indim, args.c_edge_sca_hidden)
            self.p_x_vec_embed = VNL(args.p_x_vec_indim, args.c_x_vec_hidden, leaky_relu=False)
            self.p_edge_vec_embed = VNL(args.p_edge_vec_indim, args.c_edge_vec_hidden, leaky_relu=False)


        # cycle
        self.x_gate = GVGateResidue(args.c_x_sca_hidden, args.c_x_vec_hidden, full_gate=True)
        self.edge_gate = GVGateResidue(args.c_edge_sca_hidden, args.c_edge_vec_hidden, full_gate=True)

        # for additional tasks
        self.pred_CB_layer = torch.nn.Sequential(
            GVP(args.c_x_sca_hidden, args.c_x_vec_hidden, args.c_x_sca_hidden, args.c_x_vec_hidden),
            GVL(args.c_x_sca_hidden, args.c_x_vec_hidden, args.c_x_sca_hidden, 1)
        )
        self.pred_tor_layer = torch.nn.Sequential(
            torch.nn.Linear(args.c_x_sca_hidden + args.c_x_vec_hidden,
                            args.c_x_sca_hidden + args.c_x_vec_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(args.c_x_sca_hidden + args.c_x_vec_hidden, 8),
        )
        self.pred_aff_layer = torch.nn.Sequential(
            torch.nn.Linear(args.c_x_sca_hidden + args.c_x_vec_hidden + args.c_edge_sca_hidden + args.c_edge_vec_hidden,
                            args.c_x_sca_hidden + args.c_x_vec_hidden + args.c_edge_sca_hidden + args.c_edge_vec_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(args.c_x_sca_hidden + args.c_x_vec_hidden + args.c_edge_sca_hidden + args.c_edge_vec_hidden, 1)
        )

    def init_param_with_save(self, param_path):
        if isinstance(param_path, str) and os.path.isfile(param_path):
            chk = torch.load(param_path, map_location='cpu')
        else:
            chk = load_FlexPose(param_path)
        self.init_param(chk['args'])
        self.load_state_dict(chk['model_state_dict'], strict=True)
        del chk

    def pred_label(self, complex_graph):
        # ligand coor
        l_coor_pred = rearrange(complex_graph.coor_hidden, 'b n h c -> (b n) h c')[complex_graph.ligand_node_loc_in_complex_flat]

        # CA
        CA_pred = rearrange(complex_graph.coor_hidden, 'b n h c -> (b n) h c')[complex_graph.p_partial_select_mask]

        # CB
        CB_pred = rearrange(complex_graph.coor + self.pred_CB_layer(complex_graph.x_sca_vec)[1].squeeze(-2),
                            'b n c -> (b n) c')[complex_graph.p_partial_select_mask]

        # aff
        x_sca_vec_cat = torch.cat([complex_graph.x_sca, complex_graph.x_vec.norm(p=2, dim=-1)], dim=-1)
        x_pooling = (x_sca_vec_cat * complex_graph.x_mask.float().unsqueeze(-1)).sum(dim=-2) / complex_graph.x_mask.float().sum(dim=-1, keepdims=True)
        edge_sca_vec_cat = torch.cat([complex_graph.edge_sca, complex_graph.edge_vec.norm(p=2, dim=-1)], dim=-1)
        edge_pooling = torch.einsum('b i j d -> b d', edge_sca_vec_cat * complex_graph.edge_mask.float().unsqueeze(-1)) / \
                       torch.einsum('b i j -> b', complex_graph.edge_mask.float()).unsqueeze(-1)
        x_edge_pooling = torch.cat([x_pooling, edge_pooling], dim=-1)
        aff_pred = self.pred_aff_layer(x_edge_pooling).squeeze(dim=-1)
        if not self.training:
            aff_pred = F.relu(aff_pred)

        # tor
        x_sca_vec_cat = rearrange(x_sca_vec_cat, 'b n d -> (b n) d')[complex_graph.p_partial_select_mask]
        # x_sca_vec_cat = torch.cat([x_sca_vec_cat, complex_graph.sc_in_partial_select], dim=-1)
        SC_pred = rearrange(self.pred_tor_layer(x_sca_vec_cat), '... (m s) -> ... m s', s=2)
        # if not self.training:
        #     SC_pred = SC_pred.clamp(min=-1, max=1)

        return (l_coor_pred, CA_pred, CB_pred, aff_pred, SC_pred)

    @torch.no_grad()
    def infer(self, complex_graph):
        complex_graph = self.init_embed(complex_graph)
        complex_graph = self.run_cycling(complex_graph)

        # ligand coor
        coor_pred = complex_graph.coor_hidden

        CB_pred = complex_graph.coor + self.pred_CB_layer(complex_graph.x_sca_vec)[1].squeeze(-2)

        # aff
        x_sca_vec_cat = torch.cat([complex_graph.x_sca, complex_graph.x_vec.norm(p=2, dim=-1)], dim=-1)
        x_pooling = (x_sca_vec_cat * complex_graph.x_mask.float().unsqueeze(-1)).sum(
            dim=-2) / complex_graph.x_mask.float().sum(dim=-1, keepdims=True)
        edge_sca_vec_cat = torch.cat([complex_graph.edge_sca, complex_graph.edge_vec.norm(p=2, dim=-1)], dim=-1)
        edge_pooling = torch.einsum('b i j d -> b d',
                                    edge_sca_vec_cat * complex_graph.edge_mask.float().unsqueeze(-1)) / \
                       torch.einsum('b i j -> b', complex_graph.edge_mask.float()).unsqueeze(-1)
        x_edge_pooling = torch.cat([x_pooling, edge_pooling], dim=-1)
        aff_pred = self.pred_aff_layer(x_edge_pooling).squeeze(dim=-1)
        aff_pred = F.relu(aff_pred)

        # tor
        SC_pred = rearrange(self.pred_tor_layer(x_sca_vec_cat), '... (m s) -> ... m s', s=2)

        return (coor_pred, CB_pred, aff_pred, SC_pred)

    def load_encoder(self, args):
        if 'pretrain_protein_encoder' in args.__dict__.keys():
            if isinstance(args.pretrain_protein_encoder, str) and os.path.isfile(args.pretrain_protein_encoder):
                print('Loading pre-trained protein encoder ...')
                p_param = torch.load(args.pretrain_protein_encoder, map_location='cpu')
            else:
                p_param = load_pretrained_protein_encoder(args.pretrain_protein_encoder)
            self.p_encoder.load_state_dict(p_param['model_state_dict'], strict=True)
            del p_param
        else:
            print('Skip loading pre-trained protein encoder parameters')

        if 'pretrain_ligand_encoder' in args.__dict__.keys():
            if isinstance(args.pretrain_ligand_encoder, str) and os.path.isfile(args.pretrain_ligand_encoder):
                print('Loading pre-trained ligand encoder ...')
                l_param = torch.load(args.pretrain_ligand_encoder, map_location='cpu')
            else:
                l_param = load_pretrained_ligand_encoder(args.pretrain_ligand_encoder)
            self.l_feat_encoder.load_state_dict(l_param['model_state_dict'], strict=True)
            del l_param
        else:
            print('Skip loading pre-trained ligand encoder parameters')

    def pretrain(self, complex_graph):
        # pretrain
        cur_state = self.training
        self.train(False)
        with torch.no_grad():
            complex_graph = self.p_encoder(complex_graph)
            complex_graph = self.l_feat_encoder(complex_graph)
        self.train(cur_state)

        complex_graph.p_x_sca_vec_pretrained = complex_graph.p_x_sca_vec
        complex_graph.p_edge_sca_vec_pretrained = complex_graph.p_edge_sca_vec
        complex_graph.l_x_sca_pretrained = complex_graph.l_x_sca
        complex_graph.l_edge_sca_pretrained = complex_graph.l_edge_sca
        return complex_graph

    def init_embed(self, complex_graph):
        if self.use_pretrain:
            if not self.use_pregen_data:
                complex_graph = self.pretrain(complex_graph)

            # pocket
            p_x_sca, p_x_vec = complex_graph.p_x_sca_vec_pretrained
            p_edge_sca, p_edge_vec = complex_graph.p_edge_sca_vec_pretrained
            p_x_sca = torch.cat([p_x_sca, complex_graph.sc_in], dim=-1)  # explicit torsion input
            p_x_sca = self.p_x_sca_embed(p_x_sca)
            if self.p_extra_embed:
                p_edge_sca = self.p_edge_sca_embed(p_edge_sca)
                p_x_vec = self.p_x_vec_embed(p_x_vec)
                p_edge_vec = self.p_edge_vec_embed(p_edge_vec)

            # ligand
            l_x_sca = complex_graph.l_x_sca_pretrained
            l_edge_sca = complex_graph.l_edge_sca_pretrained
            if self.l_extra_embed:
                l_x_sca = self.l_x_sca_embed(l_x_sca)
            if self.add_l_dismap:
                l_edge_sca = self.l_edge_sca_embed(torch.cat([l_edge_sca, complex_graph.l_dismap.unsqueeze(-1)], dim=-1))
            else:
                l_edge_sca = self.l_edge_sca_embed(l_edge_sca)
            l_x_vec = self.l_x_vec_embed(complex_graph.l_x_vec_init)
            l_edge_vec = self.l_edge_vec_embed(complex_graph.l_edge_vec_init)
        else:
            # pocket
            # explicit torsion input
            p_x_sca = torch.cat([complex_graph.p_x_sca_init, complex_graph.sc_in], dim=-1)
            p_x_sca = self.p_x_sca_embed(p_x_sca)
            p_edge_sca = self.p_edge_sca_embed(complex_graph.p_edge_sca_init)
            p_x_vec = self.p_x_vec_embed(complex_graph.p_x_vec_init)
            p_edge_vec = self.p_edge_vec_embed(complex_graph.p_edge_vec_init)

            # ligand
            l_x_sca = self.l_x_sca_embed(complex_graph.l_x_sca_init)
            if self.add_l_dismap:
                l_edge_sca = self.l_edge_sca_embed(torch.cat([complex_graph.l_edge_sca_init, complex_graph.l_dismap.unsqueeze(-1)], dim=-1))
            else:
                l_edge_sca = self.l_edge_sca_embed(complex_graph.l_edge_sca_init)
            l_x_vec = self.l_x_vec_embed(complex_graph.l_x_vec_init)
            l_edge_vec = self.l_edge_vec_embed(complex_graph.l_edge_vec_init)

        # merge
        complex_graph.x_sca_init = torch.cat([p_x_sca, l_x_sca], dim=1)
        complex_graph.x_vec_init = torch.cat([p_x_vec, l_x_vec], dim=1)
        complex_graph.x_sca_vec_init = (complex_graph.x_sca_init, complex_graph.x_vec_init)
        complex_graph.x_mask = torch.cat([complex_graph.p_x_mask, complex_graph.l_x_mask], dim=1)
        complex_graph.edge_sca_init = self.cat_edge(p_edge_sca, l_edge_sca)
        complex_graph.edge_vec_init = self.cat_edge(p_edge_vec, l_edge_vec)
        complex_graph.edge_sca_vec_init = (complex_graph.edge_sca_init, complex_graph.edge_vec_init)
        complex_graph.edge_mask = self.cat_edge(complex_graph.p_edge_mask, complex_graph.l_edge_mask)
        complex_graph.coor_init = torch.cat([complex_graph.p_coor_init, complex_graph.l_coor_init], dim=1)

        return complex_graph

    def run_single_cycle(self, complex_graph, cycle_i=0):
        if cycle_i == 0:
            complex_graph.x_sca_vec = complex_graph.x_sca_vec_init
            complex_graph.edge_sca_vec = complex_graph.edge_sca_vec_init
            complex_graph.coor = complex_graph.coor_init
        else:
            complex_graph.x_sca_vec = self.x_gate(complex_graph.x_sca_vec, complex_graph.x_sca_vec_init)
            complex_graph.edge_sca_vec = self.edge_gate(complex_graph.edge_sca_vec, complex_graph.edge_sca_vec_init)
        complex_graph = self.c_decoder(complex_graph)
        complex_graph.x_sca, complex_graph.x_vec = complex_graph.x_sca_vec
        complex_graph.edge_sca, complex_graph.edge_vec = complex_graph.edge_sca_vec
        return complex_graph

    def energy_min(self, complex_graph, loop=None, constraint=None, show_state=False):
        if self.use_min:
            coor_flat = rearrange(complex_graph.coor, 'b n c -> (b n) c')
            l_coor_pred = coor_flat[complex_graph.ligand_node_loc_in_complex_flat]
            l_coor_min = self.coor_min_object(l_coor_pred * self.coor_scale, complex_graph,
                                              loop=loop, constraint=constraint, show_state=show_state)
            coor_flat[complex_graph.ligand_node_loc_in_complex_flat] = l_coor_min / self.coor_scale
            complex_graph.coor = rearrange(coor_flat, '(b n) c -> b n c', b=complex_graph.coor.size(0))
        return complex_graph

    def cat_edge(self, edge_1, edge_2):
        d_1 = edge_1.size(1)
        d_2 = edge_2.size(1)
        if len(edge_1.size()) == 3:
            edge_1_pad = (0, d_2)
            edge_2_pad = (d_1, 0)
        elif len(edge_1.size()) == 4:
            edge_1_pad = (0, 0, 0, d_2)
            edge_2_pad = (0, 0, d_1, 0)
        elif len(edge_1.size()) == 5:
            edge_1_pad = (0, 0, 0, 0, 0, d_2)
            edge_2_pad = (0, 0, 0, 0, d_1, 0)
        else:
            assert len(edge_1.size()) in [3, 4, 5]
        edge_1 = F.pad(edge_1, edge_1_pad, 'constant', 0)
        edge_2 = F.pad(edge_2, edge_2_pad, 'constant', 0)
        edge = torch.cat([edge_1, edge_2], dim=1)
        return edge



if __name__=='__main__':
    pass