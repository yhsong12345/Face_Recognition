import torch
import torch.nn as nn
import torch.nn.functional as F

import math



class ArcFace(nn.Module):
    def __init__(self, embed_size, num_classes, scale=64, margin=0.5, easy_margin=False):
        """
        The input of this Module should be a Tensor which size is (N, embed_size), and the size of output Tensor is (N, num_classes).

        arcface_loss =-\sum^{m}_{i=1}log
                        \frac{e^{s\psi(\theta_{i,i})}}{e^{s\psi(\theta_{i,i})}+
                        \sum^{n}_{j\neq i}e^{s\cos(\theta_{j,i})}}
        \psi(\theta)=\cos(\theta+m)
        where m = margin, s = scale
        """
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(embed_size, num_classes))
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        nn.init.xavier_uniform_(self.weight)

    def forward(self, embed, label):
        """
        This Implementation is from https://github.com/ronghuaiyang/arcface-pytorch, which takes
        54.804054962005466 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080Ti.
        """

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(embed), F.normalize(self.weight.t())).clamp(-1 + 1e-7, 1 - 1e-7)
        sin_theta = torch.sqrt((1.0 - torch.pow(cos_theta, 2)).clamp(-1 + 1e-7, 1 - 1e-7))
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size(), device=embed.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.scale


        return output


# class ArcFace(nn.Module):
#     def __init__(self, embed_size=512, num_classes=10, scale=64.0, margin=0.50):
#         super(ArcFace, self).__init__()
#         self.in_features = embed_size
#         self.out_features = num_classes
#         self.s = scale
#         self.m = margin
#         self.kernel = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
#         nn.init.normal_(self.kernel, std=0.01)

#     def forward(self, embbedings, label):
#         embbedings = F.normalize(embbedings)
#         kernel_norm = F.normalize(self.kernel)
#         cos_theta = F.linear(embbedings, kernel_norm)
#         cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)  # for numerical stability
#         index = torch.where(label != -1)[0]
#         m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
#         m_hot.scatter_(1, label[index, None], self.m)

#         cos_theta.acos_()
#         cos_theta[index] += m_hot
#         cos_theta.cos_().mul_(self.s)
#         return cos_theta



######################################### Sub-Center ArcFace ###########################################


# class SubcenterArcFace(nn.Module):
#     r"""Modified implementation from https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py#L10
#         """
#
#     def __init__(self, in_features, out_features, K=3, s=64.0, m=0.50, easy_margin=False):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.K = K
#         self.weight = nn.Parameter(torch.FloatTensor(out_features * self.K, in_features))
#         nn.init.xavier_uniform_(self.weight)
#
#         self.easy_margin = easy_margin
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m
#
#
#     def forward(self, input, label):
#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         cosine = F.linear(F.normalize(input), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
#
#         if self.K > 1:
#             cosine = torch.reshape(cosine, (-1, self.out_features, self.K))
#             cosine, _ = torch.max(cosine, axis=2)
#
#         sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(-1 + 1e-7, 1 - 1e-7))
#         # cos(phi+m)
#         phi = cosine * self.cos_m - sine * self.sin_m
#
#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)
#
#         # --------------------------- convert label to one-hot ---------------------------
#         one_hot = torch.zeros(cosine.size(), device=input.device)
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + (
#                     (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
#         output *= self.s
#
#
#         return output


class SubcenterArcFace(nn.Module):
    def __init__(self, embed_size=512, num_classes=10,  K=3, scale=64.0, margin=0.50):
        super().__init__()
        self.in_features = embed_size
        self.out_features = num_classes
        self.s = scale
        self.m = margin
        self.K = K
        self.kernel = nn.Parameter(torch.FloatTensor(self.out_features * self.K, self.in_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = F.normalize(embbedings)
        kernel_norm = F.normalize(self.kernel)
        cos_theta = F.linear(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)  # for numerical stability
        cos_theta = torch.reshape(cos_theta, (-1, self.out_features, self.K))
        cos_theta, _ = torch.max(cos_theta, axis=2)
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)

        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta


#
# class RegularFace(nn.Module):
#     def __init__(self, embed_size, num_classes):
#         super().__init__()
#         self.num_classes = num_classes
#         self.embed_size = embed_size
#
#         self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
#         nn.init.xavier_uniform_(self.weight)
#
#
#     def forward(self, x):
#         weight_norm = F.normalize(self.weight, p=2, dim=1)
#         cos = torch.mm(weight_norm, weight_norm.t())
#
#         # for numerical stability
#         cos.clamp(-1, 1)
#
#         # for eliminate element w_i = w_j
#         cos_ind = cos.detach()
#
#         cos_ind.scatter(1, torch.arange(self.out_features).view(-1, 1).long(), -100)
#         _, indices = torch.max(cos_ind, dim=0)
#
#         mask = torch.zeros((self.out_features, self.out_features))
#         mask.scatter_(1, indices.view(-1, 1).long(), 1)
#
#         '''
#         ind = np.diag_indices(cos.shape[0])
#         min_ind = torch.min(cos_ind) - 1
#         cos_ind[ind[0], ind[1]] = torch.full((cos_ind.shape[0],), min_ind)
#
#         _,indices = torch.max(cos_ind, dim=0)
#         '''
#
#         exclusive_loss = torch.dot(cos.view(cos.numel()), mask.view(mask.numel())) / self.out_features
#
#         return exclusive_loss


# class ElasticArcFace(nn.Module):
#     def __init__(self, embed_size=512, num_classes=10, s=64.0, m=0.5, std=0.0125, easy_margin=False):
#         super().__init__()
#         self.embed_size = embed_size
#         self.num_classes = num_classes
#         self.easy_margin = easy_margin
#         self.s = s
#         self.m = m
#         self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
#         self.std = std
#         nn.init.xavier_uniform_(self.weight)
#
#     def forward(self, embed, label):
#         """
#         This Implementation is from https://github.com/ronghuaiyang/arcface-pytorch, which takes
#         54.804054962005466 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080Ti.
#         """
#
#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         cos_theta = F.linear(F.normalize(embed), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
#         cos_theta, _ = torch.sort(cos_theta, dim=0, descending=True)
#         sin_theta = torch.sqrt((1.0 - torch.pow(cos_theta, 2)).clamp(-1 + 1e-7, 1 - 1e-7))
#         margin = torch.normal(mean=self.m, std=self.std, size=cos_theta.size(), device=embed.device).clamp(self.m-self.std, self.m+self.std)
#         margin, _ = torch.sort(margin, dim=0, descending=False)
#         cos_m = torch.cos(margin)
#         sin_m = torch.sin(margin)
#         th = torch.cos(math.pi - margin)
#         mm = torch.sin(math.pi - margin) * margin
#         phi = cos_theta * cos_m - sin_theta * sin_m
#         if self.easy_margin:
#             phi = torch.where(cos_theta > 0, phi, cos_theta)
#         else:
#             phi = torch.where(cos_theta > th, phi, cos_theta - mm)
#         # --------------------------- convert label to one-hot ---------------------------
#         one_hot = torch.zeros(cos_theta.size(), device=embed.device)
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)  # you can use torch.where if your torch.__version__ is 0.4
#         output *= self.s
#
#         return output


class ElasticArcFace(nn.Module):
    def __init__(self, embed_size=512, num_classes=10, s=64.0, m=0.5, std=0.0125, plus=False):
        super(ElasticArcFace, self).__init__()
        self.in_features = embed_size
        self.out_features = num_classes
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.std=std
        self.plus=plus

    def forward(self, embbedings, label):
        embbedings = F.normalize(embbedings)
        kernel_norm = F.normalize(self.kernel)
        cos_theta = F.linear(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        margin = torch.normal(mean=self.m, std=self.std, size=label[index, None].size(), device=cos_theta.device)    #.clamp(self.m-self.std, self.m+self.std)
        if self.plus:
            with torch.no_grad():
                distmat = cos_theta[index, label.view(-1)].detach().clone()
                _, idicate_cosie = torch.sort(distmat, dim=0, descending=True)
                margin, _ = torch.sort(margin, dim=0)
            m_hot.scatter_(1, label[index, None], margin[idicate_cosie])
        else:
            m_hot.scatter_(1, label[index, None], margin)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta