import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import copy


class TeacherStudentNetwork(nn.Module):
    """
    TeacherStudentNetwork.
    """

    def __init__(
            self, net, alpha=0.999,
    ):
        super(TeacherStudentNetwork, self).__init__()
        self.net = net
        self.mean_net = copy.deepcopy(self.net)

        for param, param_m in zip(self.net.parameters(), self.mean_net.parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.alpha = alpha

    def forward(self, x):
        # if not self.training:
        #     return self.mean_net(x)

        results = self.net(x)

        with torch.no_grad():
            self._update_mean_net()  # update mean net
            results_m = self.mean_net(x)

        return results, results_m

    @torch.no_grad()
    def initialize_centers(self, centers, labels):
        self.net.initialize_centers(centers, labels)
        self.mean_net.initialize_centers(centers, labels)

    @torch.no_grad()
    def _update_mean_net(self):
        for param, param_m in zip(self.net.parameters(), self.mean_net.parameters()):
            param_m.data.mul_(self.alpha).add_(param.data, alpha=1 - self.alpha)


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)

        x = self.visible.layer1(x)
        x = self.visible.layer2(x)
        # x = self.visible.layer3(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)

        x = self.thermal.layer1(x)
        x = self.thermal.layer2(x)
        # x = self.thermal.layer3(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        # x = self.base.layer1(x)
        # x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class embed_net(nn.Module):
    def __init__(self, class_num, gm_pool='on', arch='resnet50', mean_net=True, alpha=0.999):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet_share = base_resnet(arch=arch)
        self.mean_net = mean_net

        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool

        # create mean network (optional)
        if self.mean_net:
            self.thermal_module = TeacherStudentNetwork(self.thermal_module, alpha=alpha)
            self.visible_module = TeacherStudentNetwork(self.visible_module, alpha=alpha)

    def forward(self, x1, x2, modal=0):
        # mutual learning network
        if modal == 0:
            x1, x1_mean = self.visible_module(x1)
            x2, x2_mean = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
            x_mean = torch.cat((x1_mean, x2_mean), 0)
        elif modal == 1:
            x, x_mean = self.visible_module(x1)
        elif modal == 2:
            x, x_mean = self.thermal_module(x2)

        x = self.base_resnet_share(x)
        x_mean = self.base_resnet_share(x_mean)

        if self.gm_pool == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)

            x_mean = x_mean.view(b, c, -1)
            p = 3.0
            x_pool_mean = (torch.mean(x_mean ** p, dim=-1) + 1e-12) ** (1 / p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

            x_pool_mean = self.avgpool(x_mean)
            x_pool_mean = x_pool_mean.view(x_pool_mean.size(0), x_pool_mean.size(1))

        feat = self.bottleneck(x_pool)
        feat_mean = self.bottleneck(x_pool_mean)

        if self.training:
            return x_pool, self.classifier(feat), x_pool_mean, self.classifier(feat_mean)
        else:
            return self.l2norm(x_pool_mean), self.l2norm(feat_mean)
