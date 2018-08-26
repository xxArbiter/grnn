import torch
import torch.nn as nn

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class GruCell(nn.Module):
    def __init__(self, opt):
        super(GruCell, self).__init__()
        self.dimFeature = opt.dimFeature    # d
        self.dimHidden = opt.dimHidden      # D
        
        self.resetGate = nn.Sequential(
            nn.Linear(self.dimHidden + self.dimFeature, self.dimHidden),
            nn.Sigmoid()
        )
        self.updateGate = nn.Sequential(
            nn.Linear(self.dimHidden + self.dimFeature, self.dimHidden),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(self.dimHidden + self.dimFeature, self.dimHidden),
            nn.Tanh()
        )
        self.output = nn.Linear(self.dimHidden, self.dimFeature)

    def forward(self, x, hState):
        i = torch.cat((hState, x), 1)
        z = self.resetGate(i)
        r = self.updateGate(i)
        jointI = torch.cat((r * hState, x), 1)
        hHat = self.transform(jointI)
        h = (1 - z) * hState + z * hHat
        o =  self.output(h)

        hState = h

        return o, hState


class Propogator(nn.Module):
    """
    Gated Propogator for GRNN
    Using GRU
    """
    def __init__(self, opt):
        super(Propogator, self).__init__()
        self.batchSize = opt.batchSize      # b
        self.nNode = opt.nNode              # n
        self.dimFeature = opt.dimFeature    # d
        self.dimHidden = opt.dimHidden      # D
        
        self.resetGate = nn.Sequential(
            nn.Linear(self.dimHidden + self.dimFeature, self.dimHidden),
            nn.Sigmoid()
        )
        self.updateGate = nn.Sequential(
            nn.Linear(self.dimHidden + self.dimFeature, self.dimHidden),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(self.dimHidden + self.dimFeature, self.dimHidden),
            nn.Tanh()
        )
        self.output = nn.Linear(self.dimHidden, self.dimFeature)

    def forward(self, x, hState, A):
        """
        x: [b, n, d]
        h: [b, D, n]
        """
        S = torch.bmm(hState, A)            # [b, D, n]
        S = S.transpose(1, 2)               # [b, n, D]

        i = torch.cat((S, x), 2)            # [b, n, D + d]
        z = self.resetGate(i)
        r = self.updateGate(i)
        jointI = torch.cat((r * S, x), 2)
        hHat = self.transform(jointI)
        h = (1 - z) * S + z * hHat          # [b, n, D]
        o =  self.output(h)                 # [b, n, 1]

        hState = h.transpose(1, 2)

        return o, hState


class GRNN(nn.Module):
    def __init__(self, opt):
        super(GRNN, self).__init__()
        self.batchSize = opt.batchSize      # b
        self.nNode = opt.nNode              # n
        self.dimFeature = opt.dimFeature    # d
        self.dimHidden = opt.dimHidden      # D
        self.interval = opt.truncate        # T

        self.propogator = Propogator(opt)

    def forward(self, x, hState, A):
        """
        x: input node features [batchSize, interval, nNode, dimFeature]
        hState: hidden state [batchSize, dimHidden, nNode]
        A: transfer matrix [batchSize, nNode, nNode]
        """
        O = torch.zeros(self.batchSize, self.interval, self.nNode, self.dimFeature).double()

        for t in range(self.interval):
            O[:, t, :, :], h = self.propogator(x[:, t, :, :], hState, A)
            hState = h

        return O, hState

