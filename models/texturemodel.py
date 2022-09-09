import torch
import kornia
import torch.nn as nn
from torchmeta.modules import MetaSequential, MetaModule, MetaLinear, MetaConv2d

from .metamodel import MetaModel


class Model(MetaModule, MetaModel):
    # as in https://github.com/henzler/neuralmaterial

    def __init__(self, cfg, mapping_layers=0, decoder_prefix=False, num_final_channels=3):
        super(Model, self).__init__()

        w, z, layers = cfg.model.w, cfg.model.z, cfg.model.layers
        self.layers = cfg.model.layers
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.down_mappings = nn.ModuleList()
        self.up_mappings = nn.ModuleList()
        self.w = w
        self.z = z
        self.device = cfg.device
        p = 'decoder.' if decoder_prefix is True else ''        # decoder_prefix must be True when used with encoder
        self.p = p

        self.init_mapping = Mapping(cfg.model.z, w, '{}init_mapping.affine_transform'.format(p), mapping_layers)

        self.adain = AdaptiveInstanceNormalization()

        for i in range(layers):
            f_in = int(min(w * (2 ** i), 256))   # calc down-size, 32-64, 64-128, 128-256, 256-256
            f_out = int(min(f_in * 2, 256))
            self.down_mappings.append(Mapping(z, f_out, '{}down_mappings.{}.affine_transform'.format(p, i), mapping_layers))
            self.down_blocks.append(StyleBlock(f_in, f_out, '{}down_blocks.{}.conv1'.format(p, i)))

        self.bottom_mapping = Mapping(z, f_out, '{}bottom_mapping.affine_transform'.format(p), mapping_layers)
        self.bottom_conv = StyleBlock(f_out, f_out, '{}bottom_conv.conv1'.format(p))

        for i in range(layers):
            f_in = int(min(w * (2 ** (layers - i)), 256))   # calc up-size: 256-256, 256-128, 128-64, 64-32
            f_out = int(min(w * (2 ** (layers - i-1)), 256))
            self.up_mappings.append(Mapping(z, f_out, '{}up_mappings.{}.affine_transform'.format(p, i), mapping_layers))
            self.up_blocks.append(StyleBlock(f_in, f_out, '{}up_blocks.{}.conv1'.format(p, i)))

        self.final = MetaSequential(MetaConv2d(f_out, num_final_channels, 1, 1, 0))
        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x, params=None, isTextureModel=True):
        # select meta-params here with .get_subdict, to avoid overhead of passing entire dict

        if isTextureModel:
            z = torch.ones((1, self.z), device=self.device)
        else:
            # x is a tuple that contains z and the image
            z, x = x[0], x[1]

        init_params = self.get_subdict(params, self.init_mapping.layername) if params else None

        mean, var = self.init_mapping(z, params=init_params)
        x = self.adain(x, mean, var)

        down = []

        for i in range(self.layers):
            mapping_params = self.get_subdict(params, self.down_mappings[i].layername) if params else None
            block_params = self.get_subdict(params, self.down_blocks[i].layername) if params else None

            mean, var = self.down_mappings[i](z, params=mapping_params)
            x = self.down_blocks[i](x, mean, var, params=block_params)
            down.append(x)
            x = self.max_pool(x)

        mapping_params = self.get_subdict(params, self.bottom_mapping.layername) if params else None
        block_params = self.get_subdict(params, self.bottom_conv.layername) if params else None

        mean, var = self.bottom_mapping(z, params=mapping_params)
        x = self.bottom_conv(x, mean, var, params=block_params)

        for j in range(self.layers):

            x = nn.functional.interpolate(x, mode='nearest', scale_factor=2)
            x = kornia.gaussian_blur2d(x, (3, 3), (0.2, 0.2), 'circular')
            skip_x = down[-(j+1)]

            mapping_params = self.get_subdict(params, self.up_mappings[j].layername) if params else None
            block_params = self.get_subdict(params, self.up_blocks[j].layername) if params else None

            mean, var = self.up_mappings[j](z, params=mapping_params)

            # should this line raise RuntimeError: The size of tensor a (28) must match the size of tensor b (30),
            # cf. https://github.com/tristandeleu/pytorch-meta/issues/138 or update torchmeta.
            # the solution is described in the git issue.
            x = self.up_blocks[j](x + skip_x, mean, var, params=block_params)

        final_params = self.get_subdict(params, key='{}final'.format(self.p)) if params else None

        x = self.final(x, params=final_params)

        if isTextureModel:
            x = torch.sigmoid(x)
        return x


class StyleBlock(MetaModule, nn.Module):
    def __init__(self, in_f, out_f, layername=None):
        super().__init__()

        # should this cicular padding for the metaconv raise an error,
        # e.g. RuntimeError: The size of tensor a (28) must match the size of tensor b (30),
        # cf. https://github.com/tristandeleu/pytorch-meta/issues/138 or update torchmeta
        self.conv1 = MetaConv2d(in_f, out_f, 3, 1, 1, padding_mode='circular')
        self.adain = AdaptiveInstanceNormalization()
        self.lrelu = nn.LeakyReLU()
        self.layername = layername

    def forward(self, x, mean, var, params=None):
        x = self.conv1(x, params)
        x = self.adain(x, mean, var)
        x = self.lrelu(x)
        return x


class Mapping(MetaModule, nn.Module):
    def __init__(self, z_size, out_size, layername=None, mapping_layers=0):
        super(Mapping, self).__init__()

        self.out_size = out_size

        self.mapping_layers = nn.ModuleList()

        for idx in range(mapping_layers):
            self.mapping_layers.append(MetaLinear(z_size, z_size))
            self.mapping_layers.append(nn.ReLU(inplace=True))

        self.affine_transform = MetaLinear(z_size, out_size * 2)
        self.affine_transform.bias.data[:out_size] = 0
        self.affine_transform.bias.data[out_size:] = 1
        self.layername = layername

    def forward(self, z, params=None):

        for layer in self.mapping_layers:
            z = layer(z, params)

        x = self.affine_transform(z, params)
        mean, var = torch.split(x, [self.out_size, self.out_size], dim=1)
        mean = mean[..., None, None]
        var = var[..., None, None]

        return mean, var


class AdaptiveInstanceNormalization(nn.Module):
    def __init__(self):
        super(AdaptiveInstanceNormalization, self).__init__()

    def forward(self, x, mean, std):
        whitened_x = torch.nn.functional.instance_norm(x)
        out = whitened_x * std + mean
        return out
