import torch
import torch.nn.functional as F
from torch_scatter import scatter_min, scatter_mean, scatter_add, scatter_softmax, scatter_max
from einops import rearrange, repeat, reduce

from model.GeoVec import *
from model.MMFF import *
from utils.common import *
from utils.pocket_data import *



class PocketLossFunction(torch.nn.Module):
    def __init__(self, args):
        super(PocketLossFunction, self).__init__()
        self.coor_scale = args.coor_scale

        self.focal_loss = FocalLoss()

        self.gamma_AAtype = args.gamma_AAtype
        self.gamma_CAnoise = args.gamma_CAnoise
        self.gamma_MCcoor = args.gamma_MCcoor
        self.gamma_SCcoor = args.gamma_SCcoor

    def forward(self, tup_pred, complex_graph, epoch=1e+3):
        AAtype_pred, MCcoor_pred, SCcoor_pred, CAnoise_pred = tup_pred

        # AAtype
        AAtype_loss = self.focal_loss(AAtype_pred, complex_graph.p_x_mask_AAtype_label)
        AAtype_loss = scatter_mean(AAtype_loss, complex_graph.p_x_mask_AAtype_scatter, dim=0).mean()

        # MCcoor
        MCcoor_loss = (MCcoor_pred - complex_graph.p_x_mask_MCcoor_label).norm(p=2, dim=-1)
        MCcoor_loss = scatter_mean(MCcoor_loss, complex_graph.p_x_mask_MCcoor_scatter, dim=0).mean()

        # SCcoor
        SCcoor_loss = (SCcoor_pred - complex_graph.p_x_mask_SCcoor_label).norm(p=2, dim=-1) * \
                      complex_graph.p_x_mask_SCcoor_label_SCmask
        SCcoor_loss = scatter_mean(SCcoor_loss, complex_graph.p_x_mask_SCcoor_scatter, dim=0).mean()

        # CAnoise
        CAnoise_loss = (CAnoise_pred - complex_graph.p_x_mask_CAnoise_label).norm(p=2, dim=-1)
        CAnoise_loss = scatter_mean(CAnoise_loss, complex_graph.p_x_mask_CAnoise_scatter, dim=0).mean()


        grad_loss = self.gamma_AAtype * AAtype_loss + self.gamma_CAnoise * CAnoise_loss + \
                        self.gamma_MCcoor * MCcoor_loss + self.gamma_SCcoor * SCcoor_loss

        eval_loss = {
            'grad_loss': grad_loss,
            'AAtype_loss': AAtype_loss,
            'MCcoor_loss': MCcoor_loss,
            'SCcoor_loss': SCcoor_loss,
            'CAnoise_loss': CAnoise_loss,
        }
        eval_loss = {k: v.detach().cpu().numpy() for k, v in eval_loss.items()}
        return grad_loss, eval_loss

    def get_angle(self, v1, v2):
        cos = (v1 * v2).sum(-1) / ((v1.norm(p=2, dim=-1) * v2.norm(p=2, dim=-1)) + 1e-6)
        arccos = cos.clamp(min=-np.pi + EPS, max=np.pi - EPS).arccos()
        return arccos


class LigandLossFunction(torch.nn.Module):
    def __init__(self, args):
        super(LigandLossFunction, self).__init__()
        self.focal_loss = FocalLoss()

        self.gamma_1 = args.gamma_1
        self.gamma_2 = args.gamma_2
        self.gamma_3 = args.gamma_3
        self.gamma_4 = args.gamma_4
        self.gamma_5 = args.gamma_5
        self.gamma_6 = args.gamma_6

    def forward(self, tup_pred, complex_graph, epoch=1e+3, args=None, no_grad=False, iter_=1e+3, rank=0):
        l_x_pred, l_edge_pred, graph_x_pos_pred, graph_x_neg_pred, \
            graph_edge_pos_pred, graph_edge_neg_pred, dismap_pred, noise_pred = tup_pred

        # for feat mask
        l_x_mask_loss = self.focal_loss(l_x_pred, complex_graph.l_x_mask_label)
        l_edge_mask_loss = self.focal_loss(l_edge_pred, complex_graph.l_edge_mask_label)
        l_x_mask_loss = scatter_mean(l_x_mask_loss, complex_graph.l_x_mask_info, dim=0).mean()
        l_edge_mask_loss = scatter_mean(l_edge_mask_loss, complex_graph.l_edge_mask_info, dim=0).mean()

        # for graph task
        graph_x_pos_loss = (self.focal_loss(
            graph_x_pos_pred, torch.ones_like(graph_x_pos_pred), bce=True
        ).squeeze(-1) * complex_graph.l_x_mask).sum(-1) / complex_graph.l_x_mask.sum(-1)
        graph_x_neg_loss = (self.focal_loss(
            graph_x_neg_pred, torch.zeros_like(graph_x_neg_pred), bce=True
        ).squeeze(-1) * complex_graph.shift_x_mask).sum(-1) / complex_graph.shift_x_mask.sum(-1)
        graph_edge_pos_loss = (self.focal_loss(
            graph_edge_pos_pred, torch.ones_like(graph_edge_pos_pred), bce=True
        ).squeeze(-1) * complex_graph.l_x_mask).sum(-1) / complex_graph.l_x_mask.sum(-1)
        graph_edge_neg_loss = (self.focal_loss(
            graph_edge_neg_pred, torch.zeros_like(graph_edge_neg_pred), bce=True
        ).squeeze(-1) * complex_graph.shift_x_mask).sum(-1) / complex_graph.shift_x_mask.sum(-1)
        graph_x_pos_loss = graph_x_pos_loss.mean()
        graph_x_neg_loss = graph_x_neg_loss.mean()
        graph_edge_pos_loss = graph_edge_pos_loss.mean()
        graph_edge_neg_loss = graph_edge_neg_loss.mean()
        graph_x_loss = graph_x_pos_loss + graph_x_neg_loss
        graph_edge_loss = graph_edge_pos_loss + graph_edge_neg_loss

        # for dismap
        dismap_loss = torch.einsum(
            'b i j -> b', (dismap_pred - complex_graph.ligand_dismap_true)**2 * complex_graph.ligand_dismap_mask
        ) / torch.einsum('b i j -> b', complex_graph.ligand_dismap_mask)
        dismap_loss = dismap_loss.mean()

        # for coor
        noise_loss = (noise_pred - complex_graph.l_coor_true_selected).norm(p=2, dim=-1)
        noise_loss = scatter_mean(noise_loss, complex_graph.l_x_coor_mask_info, dim=0).mean()

        # get final loss
        grad_loss = self.gamma_1 * l_x_mask_loss + \
                    self.gamma_2 * l_edge_mask_loss + \
                    self.gamma_3 * graph_x_loss + \
                    self.gamma_4 * graph_edge_loss + \
                    self.gamma_5 * dismap_loss + \
                    self.gamma_6 * noise_loss
        eval_loss = {
            'grad_loss': grad_loss,
            'l_x_mask_loss': l_x_mask_loss,
            'l_edge_mask_loss': l_edge_mask_loss,
            'graph_x_loss': graph_x_loss,
            'graph_edge_loss': graph_edge_loss,
            'graph_x_pos_loss': graph_x_pos_loss,
            'graph_x_neg_loss': graph_x_neg_loss,
            'graph_edge_pos_loss': graph_edge_pos_loss,
            'graph_edge_neg_loss': graph_edge_neg_loss,
            'dismap_loss': dismap_loss,
            'noise_loss': noise_loss,
        }
        eval_loss = {k: v.detach().cpu().numpy() for k, v in eval_loss.items()}
        return grad_loss, eval_loss


class LossFunction(torch.nn.Module):
    def __init__(self, args):
        super(LossFunction, self).__init__()
        self.coor_scale = args.coor_scale
        self.aff_scale = args.aff_scale

        self.gamma_l_coor = args.gamma_l_coor
        self.gamma_CA = args.gamma_CA
        self.gamma_CB = args.gamma_CB
        self.gamma_SC = args.gamma_SC
        self.gamma_aff = args.gamma_aff

        self.clamp_rate = args.clamp_rate
        self.clamp_max = args.clamp_max / args.coor_scale

    def forward(self, tup_pred, complex_graph, epoch=1e+3):
        l_coor_pred, CA_pred, CB_pred, aff_pred, SC_pred = tup_pred

        dic_loss = dict()
        eval_loss = dict()

        ################################################################################################################
        # pocket prediction
        ################################################################################################################
        # CA
        CA_loss = ((CA_pred - complex_graph.p_coor_true.unsqueeze(-2)) ** 2).sum(
            dim=-1) * complex_graph.p_CA_mask.unsqueeze(-1)
        CA_loss = scatter_mean(CA_loss, complex_graph.scatter_pocket, dim=0)
        CA_loss = (CA_loss[:, :-1].mean(dim=-1) + CA_loss[:, -1]).mean()
        dic_loss['CA_loss'] = CA_loss
        eval_loss['CA_metric'] = CA_loss

        # CB
        CB_loss = ((CB_pred - complex_graph.p_CB_coor_true) ** 2).sum(dim=-1) * complex_graph.p_CB_mask
        CB_loss = scatter_mean(CB_loss, complex_graph.scatter_pocket, dim=0).mean()
        dic_loss['CB_loss'] = CB_loss
        eval_loss['CB_metric'] = CB_loss

        # aff
        aff_loss = (((aff_pred - complex_graph.aff_true) ** 2) * complex_graph.aff_mask).mean()
        dic_loss['aff_loss'] = aff_loss
        eval_loss['aff_metric'] = aff_loss

        # SC tor  # use p_tor_vec_true or delta_p_tor_vec_true
        l2_sc_loss = ((SC_pred - complex_graph.p_tor_vec_true) ** 2).sum(dim=-1)
        l2_sc_alt_loss = ((SC_pred - complex_graph.p_tor_vec_alt_true) ** 2).sum(dim=-1)
        l2_sc_loss = torch.stack([l2_sc_loss, l2_sc_alt_loss], dim=0).min(dim=0)[0]

        pel_sc_loss = (SC_pred.norm(p=2, dim=-1) - 1).abs()

        sc_loss = l2_sc_loss + 0.01 * pel_sc_loss
        sc_loss = (sc_loss * complex_graph.p_tor_mask).sum(dim=-1) / (complex_graph.p_tor_mask.sum(dim=-1) + EPS)
        sc_loss = scatter_mean(sc_loss, complex_graph.scatter_pocket, dim=0).mean()

        dic_loss['sc_loss'] = sc_loss
        eval_loss['sc_metric'] = sc_loss

        with torch.no_grad():
            error_degree_pred = self.calc_cos(SC_pred, complex_graph.p_tor_vec_true).arccos().abs()
            error_degree_pred_alt = self.calc_cos(SC_pred, complex_graph.p_tor_vec_alt_true).arccos().abs()
            error_degree_pred = torch.stack([error_degree_pred, error_degree_pred_alt], dim=0).min(dim=0)[0]

            tor_degree_diff = (error_degree_pred * 180 / np.pi * complex_graph.p_tor_mask).sum(dim=-1) / (
                        complex_graph.p_tor_mask.sum(dim=-1) + EPS)
            tor_degree_diff = scatter_mean(tor_degree_diff, complex_graph.scatter_pocket, dim=0).mean()

            eval_loss['tor_degree_diff_metric'] = tor_degree_diff

        ################################################################################################################
        # ligand prediction
        ################################################################################################################
        # for coordinate prediction
        coor_pred = l_coor_pred  # (batch*n_atom, n_hidden_state, 3)
        coor_true = complex_graph.l_coor_true  # (batch*n_atom, 3)
        coor_pred = coor_pred[complex_graph.l_match]  # to (batch*n_atom*match, n_hidden_state, 3)
        coor_true = coor_true[complex_graph.l_nomatch]  # to (batch*n_atom*match, 3)

        coor_loss = torch.norm(coor_pred - coor_true.unsqueeze(dim=1), dim=-1,
                               p=2)  # to (batch*n_atom*match, n_hidden_state)
        if self.training:
            coor_loss_clamped = coor_loss.clamp(max=self.clamp_max)
            coor_loss = coor_loss_clamped * self.clamp_rate + coor_loss * (1 - self.clamp_rate)
        coor_loss = scatter_mean(coor_loss, complex_graph.scatter_ligand_1, dim=0)  # to (batch*match, n_hidden_state)
        coor_loss = scatter_min(coor_loss, complex_graph.scatter_ligand_2, dim=0)[0]  # to (batch, n_hidden_state)
        coor_grad_loss = coor_loss[:, -1] + coor_loss[:, :-1].mean(dim=-1)  # to (batch,)
        coor_grad_loss = coor_grad_loss.mean()

        dic_loss['l_coor_loss'] = coor_grad_loss
        eval_loss['l_coor_metric'] = coor_grad_loss

        # for RMSD
        with torch.no_grad():
            coor_pred = l_coor_pred[:, -1] * self.coor_scale
            coor_true = complex_graph.l_coor_true * self.coor_scale
            coor_pred = coor_pred[complex_graph.l_match]  # to (batch*n_atom*match, 3)
            coor_true = coor_true[complex_graph.l_nomatch]  # to (batch*n_atom*match, 3)

            coor_loss = ((coor_pred - coor_true) ** 2).sum(dim=-1)  # to (batch*n_atom*match,)
            coor_loss = scatter_add(coor_loss, complex_graph.scatter_ligand_1, dim=0)  # to (batch*match,)
            coor_loss = scatter_min(coor_loss, complex_graph.scatter_ligand_2, dim=0)[0]  # to (batch,)
            rmsd_loss = (coor_loss / complex_graph.len_ligand) ** 0.5
            rmsd_value = rmsd_loss.mean()
            rmsd_rate = (rmsd_loss < 2.0).to(rmsd_loss.dtype).mean()

            eval_loss['rmsd_value'] = rmsd_value
            eval_loss['rmsd_rate'] = rmsd_rate

        ################################################################################################################
        # sum all loss
        ################################################################################################################
        grad_loss = self.gamma_l_coor * dic_loss['l_coor_loss'] + \
                    self.gamma_CA * dic_loss['CA_loss'] + \
                    self.gamma_CB * dic_loss['CB_loss'] + \
                    self.gamma_SC * dic_loss['sc_loss'] + \
                    self.gamma_aff * dic_loss['aff_loss']

        eval_loss = {k: v.detach().cpu().numpy() for k, v in eval_loss.items()}
        return grad_loss, eval_loss

    def calc_cos(self, x1, x2):
        dot = (x1 * x2).sum(dim=-1)
        l_x1 = x1.norm(p=2, dim=-1)
        l_x2 = x2.norm(p=2, dim=-1)
        cos = dot / (l_x1 * l_x2 + EPS)
        cos = cos.clamp(min=-1 + EPS * 1e+2, max=1 - EPS * 1e+2)
        return cos


if __name__ == '__main__':
    pass