import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchmeta.modules import MetaModule, MetaLinear

from .metamodel import MetaModel


class Model(MetaModule, MetaModel):
    # as in http://www0.cs.ucl.ac.uk/staff/A.Sztrajman/webpage/publications/nbrdf2021/nbrdf.html

    def __init__(self, cfg):
        super(Model, self).__init__()

        self.fc1 = MetaLinear(in_features=6, out_features=21, bias=True)
        self.fc2 = MetaLinear(in_features=21, out_features=21, bias=True)
        self.fc3 = MetaLinear(in_features=21, out_features=3, bias=True)

        self.fc1.weight = nn.Parameter(torch.zeros((6, 21), dtype=torch.float32).uniform_(-0.05, 0.05).T,
                                       requires_grad=True)
        self.fc2.weight = nn.Parameter(torch.zeros((21, 21), dtype=torch.float32).uniform_(-0.05, 0.05).T,
                                       requires_grad=True)
        self.fc3.weight = nn.Parameter(torch.zeros((21, 3), dtype=torch.float32).uniform_(-0.05, 0.05).T,
                                       requires_grad=True)

        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x, params=None):
        x = F.relu(self.fc1(x, params=self.get_subdict(params, 'fc1') if params else None))
        x = F.relu(self.fc2(x, params=self.get_subdict(params, 'fc2') if params else None))
        x = torch.exp(self.fc3(x, params=self.get_subdict(params, 'fc3') if params else None)) - 1.0
        return x

    # save in a format that can be read for rendering with mitsuba
    def save_to_npy(self, epoch, savepath, params=None):
        statedict = self.state_dict() if params is None else params
        for k, v in statedict.items():
            segs = k.split('.')
            param_name = segs[0] if segs[-1] == 'weight' else segs[0].replace('fc', 'b')
            filename = 'ep{}_{}.npy'.format(epoch, param_name)
            weight = v.detach().cpu().numpy().T
            np.save(os.path.join(savepath, filename), weight)
