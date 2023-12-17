import torch

from model.MMFF import MMFFLoss
from utils.training_utils import TemporaryGrad


class CoorMin(torch.nn.Module):
    def __init__(self, args):
        super(CoorMin, self).__init__()
        self.MMFF_lossfunction = MMFFLoss(split_interact=False, warm=False)
        self.loop = args.MMFF_loop
        self.lr = args.MMFF_lr
        self.decay = args.MMFF_decay
        self.max_decay_step = args.MMFF_max_decay_step
        self.patience_tol_step = args.MMFF_patience_tol_step
        self.patience_tol_value = args.MMFF_patience_tol_value
        self.clip = args.MMFF_clip
        self.constraint = args.MMFF_constraint
        self.device = args.rank

    def forward(self, l_coor_pred, complex_graph, loop=None, constraint=None, show_state=False, min_type='GD'):
        with TemporaryGrad():  # i.e. torch.set_grad_enabled(True)
            coor_min = self.FF_min(l_coor_pred, complex_graph,
               loop=self.loop if isinstance(loop, type(None)) else loop,
               lr=self.lr,
               decay=self.decay, max_decay_step=self.max_decay_step,
               patience_tol_step=self.patience_tol_step, patience_tol_value=self.patience_tol_value,
               clip=self.clip,
               constraint=self.constraint if isinstance(constraint, type(None)) else constraint,
               show_state=show_state, min_type=min_type,
               )
        if torch.isnan(coor_min).any():
            return l_coor_pred
        return coor_min

    def FF_min(self, coor_pred, complex_graph,
               loop=10000, lr=5e-5,
               decay=0.5, max_decay_step=10,
               patience_tol_step=100, patience_tol_value=0,
               clip=1e+5, constraint=0, show_state=False, min_type='GD'):

        coor_pred_detach = coor_pred.detach()
        coor_paramed = torch.nn.Parameter(coor_pred.detach().clone()).to(coor_pred.device)
        if min_type == 'GD':
            optimizer = torch.optim.SGD([coor_paramed], lr=lr)
        elif min_type == 'LBFGS':
            optimizer = torch.optim.LBFGS([coor_paramed], lr=1.0, line_search_fn='strong_wolfe')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, decay)

        for step in range(loop):
            def closure():
                e = self.MMFF_lossfunction(coor_paramed, complex_graph, return_sum=True)
                e = e.sum()
                if constraint > 0:
                    e_constraint = 0.5 * ((coor_paramed - coor_pred_detach) ** 2 * constraint).sum()
                    e = e + e_constraint
                optimizer.zero_grad()
                e.backward()
                if min_type == 'GD':
                    torch.nn.utils.clip_grad_norm_([coor_paramed], clip)
                return e
            e = closure()

            if min_type == 'GD':
                optimizer.step()
            elif min_type == 'LBFGS':
                optimizer.step(closure)

            if step == 0:
                e_min = e
                patience_sum = 0
                decay_step = 0
            elif e_min - e < patience_tol_value:
                patience_sum += 1
            else:
                e_min = e
            if patience_sum >= patience_tol_step:
                if decay_step < max_decay_step:
                    scheduler.step()
                    patience_sum = 0
                decay_step += 1
            if torch.isnan(e).any():
                break

            if show_state:
                print(f"Step:{step + 1}, lr:{optimizer.param_groups[0]['lr']:.2e}, E:{e.detach().cpu().numpy():.3e}")
        coor_pred_min = coor_paramed.detach().clone().to(coor_pred.device)
        del optimizer
        del scheduler
        del coor_paramed
        torch.cuda.empty_cache()
        return coor_pred_min