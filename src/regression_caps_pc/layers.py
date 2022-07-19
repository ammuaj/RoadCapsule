"""RoadCaps layers."""

import torch
from torch.autograd import Variable

class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for _ in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)

class PrimaryCapsuleLayer(torch.nn.Module):
    """
    Primary Convolutional Capsule Layer class based on:
    https://github.com/timomernick/pytorch-capsule.
    """
    def __init__(self, in_units, in_channels, num_units, capsule_dimensions):
        super(PrimaryCapsuleLayer, self).__init__()
        """
        :param in_units: Number of input units (GCN layers).
        :param in_channels: Number of channels.
        :param num_units: Number of capsules.
        :param capsule_dimensions: Number of neurons in capsule.
        """
        self.num_units = num_units
        self.units = []
        for i in range(self.num_units):
            unit = torch.nn.Conv1d(in_channels=in_channels,
                                   out_channels=capsule_dimensions,
                                   kernel_size=(in_units, 1),
                                   stride=1,
                                   bias=True)

            self.add_module("unit_" + str(i), unit)
            self.units.append(unit)

    @staticmethod
    def squash(s):
        """
        Squash activations.
        :param s: Signal.
        :return s: Activated signal.
        """
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        """
        Forward propagation pass.
        :param x: Input features.
        :return : Primary capsule features.
        """
        u = [self.units[i](x) for i in range(self.num_units)]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_units, -1)
        return PrimaryCapsuleLayer.squash(u)

class HigherCapsuleLayer(torch.nn.Module):
    """
    Secondary Convolutional Capsule Layer class based on this repostory:
    https://github.com/timomernick/pytorch-capsule
    """
    def __init__(self, in_units, in_channels, num_units, unit_size):
        super(HigherCapsuleLayer, self).__init__()
        """
        :param in_units: Number of input units (GCN layers).
        :param in_channels: Number of channels.
        :param num_units: Number of capsules.
        :param capsule_dimensions: Number of neurons in capsule.
        """
        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.W = torch.nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))

    @staticmethod
    def squash(s):
        """
        Squash activations.
        :param s: Signal.
        :return s: Activated signal.
        """
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        """
        Forward propagation pass.
        :param x: Input features.
        :return : Capsule output.
        """
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_units, 1))

        num_iterations = 3

        for _ in range(num_iterations):
            c_ij = torch.nn.functional.softmax(b_ij, dim=2)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = HigherCapsuleLayer.squash(s_j)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)
            b_ij = b_ij + u_vj1
            # b_max = torch.max(b_ij, dim = 2, keepdim = True)
            # b_ij = b_ij / b_max.values ## values can be zero so loss would be nan
        #print(" output of second layer line: 146: ", b_ij)
        return v_j.squeeze(1)

def mse_loss(scores, target, loss_lambda):
    """
    The margin loss from the original paper. Based on:
    https://github.com/timomernick/pytorch-capsule
    :param scores: Capsule scores.
    :param target: Target groundtruth.
    :param loss_lambda: Regularization parameter.
    :return L_c: Classification loss.
    """
    #print("predict size: ", scores.size())
    scores = scores.squeeze()
    #v_mag = torch.torch.sqrt((scores**2).sum(dim=1))
    v_mag =torch.sum(scores, dim=1)
    #print("v_mag_size: ",v_mag.size())
    #print("prediction",v_mag)
    #print(target.size())
    #print(v_mag)
    #print(target)
    #print('diff ',v_mag-target)
    #L_r = torch.mean(torch.sqrt((v_mag-target)**2))
    #L_r = torch.nn.L1Loss().forward(v_mag,target)
    L_r = torch.nn.MSELoss().forward(v_mag,target)
    #print(L_r)
    #L-r = L_r+loss_lambda*torch.max((v_mag-target)**2).

    return L_r

