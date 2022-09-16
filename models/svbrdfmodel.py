import os
import sys
import torch
from torchmeta.modules import MetaModule

from .metamodel import MetaModel
from .svbrdf_renderer import Renderer
from .texturemodel import Model as Decoder
from .helpers import resnet50, RotationEncoder


class Model(MetaModule, MetaModel):

    def __init__(self, cfg):
        super(Model, self).__init__()

        if cfg.data.use_pretrained_encoder is True:
            if cfg.meta.load_checkpt is True:
                print("Warning - the provided svbrdf checkpt has been trained without encoder,"
                      " but you are loading both the encoder *and* the checkpt. This will create wrong results. "
                      "To be able to use the encoder and checkpt together, retrain the checkpt.",
                      file=sys.stderr)

            checkpt_path = os.path.join(cfg.basepath,
                                        cfg.data.pretrained_encoder_dir, 'henzler_pretrained.ckpt')
            if not os.path.exists(checkpt_path):
                res = input("Downloading pretrained encoder from Github. Continue? y/n ")
                if res != 'y': sys.exit()
                os.system('wget https://github.com/henzler/neuralmaterial/raw/master/trainings/Neuralmaterial/checkpoint/latest.ckpt')

                # discard decoder
                state = torch.load('latest.ckpt')
                state_encoder = {k: v for k, v in state.items() if 'encoder' in k}
                torch.save(state_encoder, 'latest.ckpt')

                os.system('mv latest.ckpt {}'.format(checkpt_path))

        self.has_encoder = cfg.data.use_pretrained_encoder
        self.rot_encoder = None
        self.encoder = None
        self.cfg = cfg

        if self.has_encoder:
            enc_state = torch.load(os.path.join(cfg.basepath,
                                                cfg.data.pretrained_encoder_dir, 'henzler_pretrained.ckpt'))
            self.rot_encoder = RotationEncoder().to(cfg.device)
            self.encoder = resnet50(pretrained=True, num_classes=cfg.model.z).to(cfg.device)
            state_rotencoder = {k.replace('rotation_encoder.', ''): v for k, v in enc_state.items() if 'rotation' in k}
            state_encoder = {k.replace('encoder.', ''): v for k, v in enc_state.items() if not 'rotation' in k}
            self.rot_encoder.load_state_dict(state_rotencoder)
            self.encoder.load_state_dict(state_encoder)

            # turn off grad, we do not train these bc of memory constraints
            self.rot_encoder.requires_grad_(False)
            self.encoder.requires_grad_(False)

        self.renderer = Renderer(fov=cfg.model.renderer.fov,
                                 gamma=cfg.model.renderer.gamma,
                                 attenuation=cfg.model.renderer.attenuation, device=cfg.device).to(cfg.device)
        self.render_position = self.renderer.get_position(cfg.data.size).to(cfg.device)
        self.decoder = Decoder(cfg, num_final_channels=6).to(cfg.device)

    # overwrite params() and named_params() to only train decoder when outer_optimizer calls .params()
    def parameters(self, recurse=True):
        return self.decoder.parameters(recurse)

    def named_parameters(self, prefix='', recurse=True):
        return self.decoder.named_parameters(prefix, recurse)

    # when called from abstract base class, load only decoder weights as this is what we're saving
    def load_state_dict(self, state_dict, strict=True):
        self.decoder.load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.decoder.state_dict()

    def forward(self, x, params=None):

        with torch.no_grad():
            if self.has_encoder:
                # if the encoder is used, the passed x will be a concatenation of noise and the image.
                # this is necessary in order to comply with the generic interface of the metappearance fw pass.
                image = x[:, -3:, :, :]     # [b, 3, height, width]
                x = x[:, :-3, :, :]         # [b, w, height, width]
                z, _ = self.encoder(image)
                rot = self.rot_encoder(image)
            else:
                z = torch.ones((1, self.cfg.model.z), device=self.cfg.device)
                rot = torch.zeros((1, 2), device=self.cfg.device)

        decoding = self.decoder([z, x], params=params, isTextureModel=False)
        brdf_maps = {
            'diffuse': torch.sigmoid(decoding[:, :3]),
            'specular': torch.sigmoid(decoding[:, 3:4]),
            'roughness': torch.sigmoid(decoding[:, 4:5]).clamp(0.01, 0.99),
            'normal': self.renderer.height_to_normal(torch.sigmoid(decoding[:, 5:6]))
        }
        image_out = self.renderer(brdf_maps, self.render_position, rotation_angle=rot, light_pos_shift=None)
        return image_out

