import torch

class mmd_repulsive:
    def __init__(self, score_gen, score_data, batch_size, do_summary=False, repulsive_weights=None, dis_penalty=None, dis_scale=None):
        self.score_gen = score_gen
        self.score_data = score_data
        self.batch_size = batch_size
        self.do_summary = do_summary
        self.repulsive_weights = repulsive_weights
        self.dis_penalty = dis_penalty
        self.dis_scale = dis_scale
        self.loss_gen = None
        self.loss_dis = None

    def get_squared_dist(self, x, y=None, scale=None, z_score=False, mode='xxxyyy', do_summary=False, scope_prefix=''):
        if len(x.shape) > 2:
            raise AttributeError('get_dist: Input must be a matrix.')

        if y is None:
            mode = 'xx'

        if z_score:
            if y is None:
                mu = torch.mean(x, dim=0, keepdim=True)
                x = x - mu
            else:
                mu = torch.mean(torch.cat((x, y), dim=0), dim=0, keepdim=True)
                x = x - mu
                y = y - mu

        if mode in ['xx', 'xxxy', 'xxxyyy']:
            if scale is None:
                xxt = torch.matmul(x, x.t())
            else:
                xxt = torch.matmul(x * scale, x.t())
            dx = torch.diag(xxt)
            dist_xx = torch.clamp(torch.unsqueeze(dx, 1) - 2.0 * xxt + torch.unsqueeze(dx, 0), min=0.0)

            if mode == 'xx':
                return dist_xx
            elif mode == 'xxxy':
                if scale is None:
                    xyt = torch.matmul(x, y.t())
                    dy = torch.sum(y * y, dim=1)
                else:
                    xyt = torch.matmul(x * scale, y.t())
                    dy = torch.sum(y * scale * y, dim=1)
                dist_xy = torch.clamp(torch.unsqueeze(dx, 1) - 2.0 * xyt + torch.unsqueeze(dy, 0), min=0.0)
                return dist_xx, dist_xy
            elif mode == 'xxxyyy':
                if scale is None:
                    xyt = torch.matmul(x, y.t())
                    yyt = torch.matmul(y, y.t())
                else:
                    xyt = torch.matmul(x * scale, y.t())
                    yyt = torch.matmul(y * scale, y.t())
                dy = torch.diag(yyt)
                dist_xy = torch.clamp(torch.unsqueeze(dx, 1) - 2.0 * xyt + torch.unsqueeze(dy, 0), min=0.0)
                dist_yy = torch.clamp(torch.unsqueeze(dy, 1) - 2.0 * yyt + torch.unsqueeze(dy, 0), min=0.0)
                return dist_xx, dist_xy, dist_yy

        elif mode == 'xy':
            if scale is None:
                dx = torch.sum(x * x, dim=1)
                dy = torch.sum(y * y, dim=1)
                xyt = torch.matmul(x, y.t())
            else:
                dx = torch.sum(x * scale * x, dim=1)
                dy = torch.sum(y * scale * y, dim=1)
                xyt = torch.matmul(x * scale, y.t())
            dist_xy = torch.clamp(torch.unsqueeze(dx, 1) - 2.0 * xyt + torch.unsqueeze(dy, 0), min=0.0)
            return dist_xy
        else:
            raise AttributeError('Mode {} not supported'.format(mode))

    def matrix_mean_wo_diagonal(self, matrix, num_row, num_col=None):
        if num_col is None:
            num_col = num_row
        total_sum = torch.sum(matrix)
        diag_sum = torch.sum(torch.diag(matrix))
        num_elements = num_row * num_col - min(num_row, num_col)
        mean_wo_diag = (total_sum - diag_sum) / num_elements
        return mean_wo_diag

    def mmd_g_bounded(self, dist_xx, dist_xy, dist_yy, batch_size, sigma=1.0, var_target=None, upper_bound=None, lower_bound=None, name='mmd', do_summary=False, scope_prefix='', custom_weights=None):
        k_xx = torch.exp(-dist_xx / (2.0 * sigma ** 2))
        k_yy = torch.exp(-dist_yy / (2.0 * sigma ** 2))
        k_xy = torch.exp(-dist_xy / (2.0 * sigma ** 2))

        k_xx_b = torch.exp(-torch.clamp(dist_xx, min=lower_bound) / (2.0 * sigma ** 2))
        if custom_weights[0] > 0:
            k_xy_b = torch.exp(-torch.clamp(dist_xy, max=upper_bound) / (2.0 * sigma ** 2))
        else:
            k_xy_b = k_xy
        if custom_weights[1] > 0:
            k_yy_b = torch.exp(-torch.clamp(dist_yy, min=lower_bound) / (2.0 * sigma ** 2))
        else:
            k_yy_b = torch.exp(-torch.clamp(dist_yy, max=upper_bound) / (2.0 * sigma ** 2))

        m = batch_size
        e_kxx = self.matrix_mean_wo_diagonal(k_xx, m)
        e_kxy = self.matrix_mean_wo_diagonal(k_xy, m)
        e_kyy = self.matrix_mean_wo_diagonal(k_yy, m)
        e_kxx_b = self.matrix_mean_wo_diagonal(k_xx_b, m)
        e_kyy_b = self.matrix_mean_wo_diagonal(k_yy_b, m)
        e_kxy_b = self.matrix_mean_wo_diagonal(k_xy_b, m) if custom_weights[0] < 0 else e_kxy

        if do_summary:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            writer.add_scalar(scope_prefix + name + '/kxx', e_kxx)
            writer.add_scalar(scope_prefix + name + '/kyy', e_kyy)
            writer.add_scalar(scope_prefix + name + '/kxy', e_kxy)
            writer.add_scalar(scope_prefix + name + '/kxx_b', e_kxx_b)
            writer.add_scalar(scope_prefix + name + '/kyy_b', e_kyy_b)
            if custom_weights[0] > 0:
                writer.add_scalar(scope_prefix + name + '/kxy_b', e_kxy_b)

        if var_target is None:
            if custom_weights is None:
                mmd = e_kxx + e_kyy - 2.0 * e_kxy
                return mmd
            else:
                assert custom_weights[0] - custom_weights[1] == 1.0, 'w[0]-w[1] must be 1'
                mmd1 = e_kxx + e_kyy - 2.0 * e_kxy
                mmd2 = custom_weights[0] * e_kxy_b - e_kxx_b - custom_weights[1] * e_kyy_b
                print("xxx",mmd1)
                print("yyy",mmd1)
                return mmd1, mmd2
        else:
            mmd = e_kxx + e_kyy - 2.0 * e_kxy
            var = e_kxx + e_kyy + 2.0 * e_kxy
            loss_sigma = (var - var_target) ** 2
            if do_summary:
                writer.add_scalar(scope_prefix + name + '/loss_sigma', loss_sigma)

            return mmd, loss_sigma

    def _repulsive_mmd_g_bounded_(self):
        dist_gg, dist_gd, dist_dd = self.get_squared_dist(self.score_gen, self.score_data, z_score=False, do_summary=self.do_summary)
        self.loss_gen, self.loss_dis = self.mmd_g_bounded(
            dist_gg, dist_gd, dist_dd, self.batch_size, sigma=1.0, lower_bound=0.25, upper_bound=4.0,
            name='mmd_g', do_summary=self.do_summary, scope_prefix='', custom_weights=self.repulsive_weights)
        if self.dis_penalty is not None:
            self.loss_dis = self.loss_dis + self.dis_penalty
            if self.do_summary:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter()
                writer.add_scalar('GANLoss/dis_penalty', self.dis_penalty)
        if self.dis_scale is not None:
            self.loss_dis = self.loss_dis * self.dis_scale
            if self.do_summary:
                writer.add_scalar('GANLoss/dis_scale', self.dis_scale)
