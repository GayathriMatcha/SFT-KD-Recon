import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# **KD_METHODS** #
'''
(KD) - Distilling the Knowledge in a Neural Network
(FitNet) - Fitnets: hints for thin deep nets
(AT) - Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
(SP) - Similarity-Preserving Knowledge Distillation
(CC) - Correlation Congruence for Knowledge Distillation
(VID) - Variational Information Distillation for Knowledge Transfer
(RKD) - Relational Knowledge Distillation
(PKT) - Probabilistic Knowledge Transfer for deep representation learning
(AB) - Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
(FT) - Paraphrasing Complex Network: Network Compression via Factor Transfer
(FSP) - A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
(NST) - Like what you like: knowledge distill via neuron selectivity transfer

'''
#2.###################################################$$ MAIN_KD $$######################################################################    
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T=1):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=None) * (self.T**2) / y_s.shape[0]
        return loss
#3.####################################################$$ FITNETS $$################################################################
class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss
    
#4.#################################################### ATTENTION TRANSFER ################################################################
    
class Attention(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""
    def __init__(self, p=2):
        super(Attention, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return [self.at_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))
    
#5.#################################################### SIMILARITY PRESERVING ########################################################

class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):
        return [self.similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss

#6.#################################################### PKT ################################################################

class PKT(nn.Module):
    """Probabilistic Knowledge Transfer for deep representation learning
    Code from author: https://github.com/passalis/probabilistic_kt"""
    def __init__(self):
        super(PKT, self).__init__()

    def forward(self, f_s, f_t):
        return self.cosine_similarity_loss(f_s, f_t)

    @staticmethod
    def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
        # Normalize each vector by its norm
        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + eps)
        output_net[output_net != output_net] = 0

        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + eps)
        target_net[target_net != target_net] = 0

        # Calculate the cosine similarity
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

        # Scale cosine similarity to 0..1
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        # Transform them into probabilities
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))

        return loss
    
#7.################################################ Activation Boundaries #################################################################

class ABLoss(nn.Module):
    """Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons
    code: https://github.com/bhheo/AB_distillation
    """
    def __init__(self, feat_num, margin=1.0):
        super(ABLoss, self).__init__()
        self.w = [2**(i-feat_num+1) for i in range(feat_num)]
        self.margin = margin

    def forward(self, g_s, g_t):
        bsz = g_s[0].shape[0]
        losses = [self.criterion_alternative_l2(s, t) for s, t in zip(g_s, g_t)]
        losses = [w * l for w, l in zip(self.w, losses)]
        # loss = sum(losses) / bsz
        # loss = loss / 1000 * 3
        losses = [l / bsz for l in losses]
        losses = [l / 1000 * 3 for l in losses]
        return losses

    def criterion_alternative_l2(self, source, target):
        loss = ((source + self.margin) ** 2 * ((source > -self.margin) & (target <= 0)).float() +
                (source - self.margin) ** 2 * ((source <= self.margin) & (target > 0)).float())
        return torch.abs(loss).sum()
    
#8.############################################### Correlation Congruence ##################################################################

class Correlation(nn.Module):
    """Correlation Congruence for Knowledge Distillation, ICCV 2019.
    The authors nicely shared the code with me. I restructured their code to be 
    compatible with my running framework. Credits go to the original author"""
    def __init__(self):
        super(Correlation, self).__init__()

    def forward(self, f_s, f_t):
        delta = torch.abs(f_s - f_t)
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss


#9.#############################################$$ Flow of solution Procedure $$#########################################################

class FSP(nn.Module):
    """A Gift from Knowledge Distillation:
    Fast Optimization, Network Minimization and Transfer Learning"""
    def __init__(self, s_shapes, t_shapes):
        super(FSP, self).__init__()
#         assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
#         s_c = [s[1] for s in s_shapes]
#         t_c = [t[1] for t in t_shapes]
#         if np.any(np.asarray(s_c) != np.asarray(t_c)):
#             raise ValueError('num of channels not equal (error in FSP)')
        if s_shapes != t_shapes:
            raise ValueError('num of channels not equal (error in FSP)')

    def forward(self, g_s, g_t):
        s_fsp = self.compute_fsp(g_s)
        t_fsp = self.compute_fsp(g_t)
        loss_group = [self.compute_loss(s, t) for s, t in zip(s_fsp, t_fsp)]
        return loss_group

    @staticmethod
    def compute_loss(s, t):
        return (s - t).pow(2).mean()

    @staticmethod
    def compute_fsp(g):
        fsp_list = []
        for i in range(len(g) - 1):
            bot, top = g[i], g[i + 1]
            b_H, t_H = bot.shape[2], top.shape[2]
            if b_H > t_H:
                bot = F.adaptive_avg_pool2d(bot, (t_H, t_H))
            elif b_H < t_H:
                top = F.adaptive_avg_pool2d(top, (b_H, b_H))
            else:
                pass
            bot = bot.unsqueeze(1)
            top = top.unsqueeze(2)
            bot = bot.view(bot.shape[0], bot.shape[1], bot.shape[2], -1)
            top = top.view(top.shape[0], top.shape[1], top.shape[2], -1)

            fsp = (bot * top).mean(-1)
            fsp_list.append(fsp)
        return fsp_list
    
#10.####################################################$$ Factor Transfer $$ #############################################################

class FactorTransfer(nn.Module):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer, NeurIPS 2018"""
    def __init__(self, p1=2, p2=1):
        super(FactorTransfer, self).__init__()
        self.p1 = p1
        self.p2 = p2

    def forward(self, f_s, f_t):
        return self.factor_loss(f_s, f_t)

    def factor_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        if self.p2 == 1:
            return (self.factor(f_s) - self.factor(f_t)).abs().mean()
        else:
            return (self.factor(f_s) - self.factor(f_t)).pow(self.p2).mean()

    def factor(self, f):
        return F.normalize(f.pow(self.p1).mean(1).view(f.size(0), -1))
    
#11.############################################# $$ Variational Information distillation $$ ##############################################

class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation"""
    def __init__(self,
                 num_input_channels,
                 num_mid_channel,
                 num_target_channels,
                 init_pred_var=5.0,
                 eps=1e-5):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var-eps)-1.0) * torch.ones(num_target_channels)
            )
        self.eps = eps

    def forward(self, input, target):
        # pool for dimentsion match
        s_H, t_H = input.shape[2], target.shape[2]
        if s_H > t_H:
            input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        elif s_H < t_H:
            target = F.adaptive_avg_pool2d(target, (s_H, s_H))
        else:
            pass
        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0+torch.exp(self.log_scale))+self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5*(
            (pred_mean-target)**2/pred_var+torch.log(pred_var)
            )
        loss = torch.mean(neg_log_prob)
        return loss
    
#12.######################################################################################################################################

class KDSVD(nn.Module):
    """
    Self-supervised Knowledge Distillation using Singular Value Decomposition
    original Tensorflow code: https://github.com/sseung0703/SSKD_SVD
    """
    def __init__(self, k=1):
        super(KDSVD, self).__init__()
        self.k = k

    def forward(self, g_s, g_t):
        v_sb = None
        v_tb = None
        losses = []
        for i, f_s, f_t in zip(range(len(g_s)), g_s, g_t):

            u_t, s_t, v_t = self.svd(f_t, self.k)
            u_s, s_s, v_s = self.svd(f_s, self.k + 3)
            v_s, v_t = self.align_rsv(v_s, v_t)
            s_t = s_t.unsqueeze(1)
            v_t = v_t * s_t
            v_s = v_s * s_t

            if i > 0:
                s_rbf = torch.exp(-(v_s.unsqueeze(2) - v_sb.unsqueeze(1)).pow(2) / 8)
                t_rbf = torch.exp(-(v_t.unsqueeze(2) - v_tb.unsqueeze(1)).pow(2) / 8)

                l2loss = (s_rbf - t_rbf.detach()).pow(2)
                l2loss = torch.where(torch.isfinite(l2loss), l2loss, torch.zeros_like(l2loss))
                losses.append(l2loss.sum())

            v_tb = v_t
            v_sb = v_s

        bsz = g_s[0].shape[0]
        losses = [l / bsz for l in losses]
        return losses

    def svd(self, feat, n=1):
        size = feat.shape
        assert len(size) == 4

        x = feat.view(-1, size[1], size[2] * size[2]).transpose(-2, -1)
        u, s, v = torch.svd(x)

        u = self.removenan(u)
        s = self.removenan(s)
        v = self.removenan(v)

        if n > 0:
            u = F.normalize(u[:, :, :n], dim=1)
            s = F.normalize(s[:, :n], dim=1)
            v = F.normalize(v[:, :, :n], dim=1)

        return u, s, v

    @staticmethod
    def removenan(x):
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        return x

    @staticmethod
    def align_rsv(a, b):
        cosine = torch.matmul(a.transpose(-2, -1), b)
        max_abs_cosine, _ = torch.max(torch.abs(cosine), 1, keepdim=True)
        mask = torch.where(torch.eq(max_abs_cosine, torch.abs(cosine)),
                           torch.sign(cosine), torch.zeros_like(cosine))
        a = torch.matmul(a, mask)
        return a, b
    
#13.########################################################################################################################################

class NSTLoss(nn.Module):
    """like what you like: knowledge distill via neuron selectivity transfer"""
    def __init__(self):
        super(NSTLoss, self).__init__()
        pass

    def forward(self, g_s, g_t):
        return [self.nst_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]

    def nst_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass

        f_s = f_s.view(f_s.shape[0], f_s.shape[1], -1)
        f_s = F.normalize(f_s, dim=2)
        f_t = f_t.view(f_t.shape[0], f_t.shape[1], -1)
        f_t = F.normalize(f_t, dim=2)

        # set full_loss as False to avoid unnecessary computation
        full_loss = True
        if full_loss:
            return (self.poly_kernel(f_t, f_t).mean().detach() + self.poly_kernel(f_s, f_s).mean()
                    - 2 * self.poly_kernel(f_s, f_t).mean())
        else:
            return self.poly_kernel(f_s, f_s).mean() - 2 * self.poly_kernel(f_s, f_t).mean()

    def poly_kernel(self, a, b):
        a = a.unsqueeze(1)
        b = b.unsqueeze(2)
        res = (a * b).sum(-1).pow(2)
        return res
    
###########################################################################################################################################

class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""
    def __init__(self, w_d=25, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)

        # RKD distance loss
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD Angle loss
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        loss = self.w_d * loss_d + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res