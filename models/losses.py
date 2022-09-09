import torch
import kornia
from .helpers import VGGFeatures, GramMatrix, grid_sample


class BRDFLoss(torch.nn.Module):

    def __init__(self):
        super(BRDFLoss, self).__init__()

    def forward(self, model_input, y_pred, y_true):
        rgb_pred = brdf_to_rgb(model_input, y_pred)
        rgb_true = brdf_to_rgb(model_input, y_true)
        return torch.mean(torch.abs(torch.log(1 + rgb_true) - torch.log(1 + rgb_pred)))


class TextureLoss(torch.nn.Module):

    def __init__(self, device='cuda'):
        super(TextureLoss, self).__init__()

        self.vgg = VGGFeatures().to(device)
        self.gram_loss = GramLoss().to(device)

    def forward(self, model_input, y_pred, y_true):
        stats_pred = self.vgg(y_pred)
        stats_true = self.vgg(y_true)
        gram_loss = self.gram_loss(stats_pred, stats_true)
        return gram_loss


class SVBRDFLoss(torch.nn.Module):

    def __init__(self, cfg, device='cuda'):
        super(SVBRDFLoss, self).__init__()

        self.cfg = cfg

        self.vgg = VGGFeatures().to(device)
        self.gram_loss = GramLoss().to(device)
        self.vggps_loss = VGGPSLoss().to(device)

        self.gram_weight = cfg.model.loss.gram
        self.vggps_weight = cfg.model.loss.vggps

    def get_crops(self, image_in, image_out):

        bs, c, h, w = image_in.size()
        resample_size_h = self.cfg.data.size[0]
        resample_size_w = self.cfg.data.size[1]

        rand = torch.rand((1, )).item()
        start = self.cfg.data.crop[0]
        end = self.cfg.data.crop[1]
        zoom_factor = rand * (end - start) + start

        res = self.cfg.data.size[0] * zoom_factor
        downscale = res / resample_size_h
        sigma = 2 * downscale / 6.0

        if zoom_factor > 1:
            image_in = kornia.filters.gaussian_blur2d(image_in, (5, 5), (sigma, sigma))
            image_out = kornia.filters.gaussian_blur2d(image_out, (5, 5), (sigma, sigma))

        grid = kornia.create_meshgrid(resample_size_h, resample_size_w, normalized_coordinates=True,
                                      device=self.cfg.device).expand(bs, resample_size_h, resample_size_w, 2)

        grid = grid + 1 + torch.rand((1,)).item() * 2
        grid = (grid * zoom_factor) % 4.0
        grid = torch.where(grid > 2, 4 - grid, grid)
        grid = grid - 1

        crops_in = grid_sample(image_in, grid=grid)
        crops_out = grid_sample(image_out, grid=grid)
        return crops_in, crops_out

    def forward(self, model_input, y_pred, y_true):
        crops_in, crops_out = self.get_crops(y_true, y_pred)
        crops_in_vgg = self.vgg(crops_in)
        crops_out_vgg = self.vgg(crops_out)
        gram_loss = self.gram_loss(crops_out_vgg, crops_in_vgg)
        vggps_loss = self.vggps_loss(crops_out_vgg, crops_in_vgg)
        weighted_loss = gram_loss * self.gram_weight + vggps_loss * self.vggps_weight
        return weighted_loss.mean()


class VGGPSLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, x, y):
        loss = torch.tensor(0.0, device=x[0].device)

        features_out = x
        features_gt = y

        for idx, (feature_out, feature_gt) in enumerate(zip(features_out, features_gt)):
            x_power = torch.abs(torch.fft.fftn(feature_out, dim=[2, 3]))
            y_power = torch.abs(torch.fft.fftn(feature_gt, dim=[2, 3]))
            loss += self.l1_loss(x_power, y_power).sum()
        return loss


class GramLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.gram_matrix = GramMatrix()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, x, y):
        loss = torch.tensor(0.0, device=x[0].device)

        input_features = x
        output_features = y

        for idx, (input_feature, output_feature) in enumerate(zip(input_features, output_features)):
            gram_out = self.gram_matrix(output_feature)
            gram_in = self.gram_matrix(input_feature)
            loss += self.l1_loss(gram_out, gram_in).mean()

        return loss


def brdf_to_rgb(rvectors, brdf):
    hx = torch.reshape(rvectors[:, 0], (-1, 1))
    hy = torch.reshape(rvectors[:, 1], (-1, 1))
    hz = torch.reshape(rvectors[:, 2], (-1, 1))
    dx = torch.reshape(rvectors[:, 3], (-1, 1))
    dy = torch.reshape(rvectors[:, 4], (-1, 1))
    dz = torch.reshape(rvectors[:, 5], (-1, 1))

    theta_h = torch.atan2(torch.sqrt(hx ** 2 + hy ** 2), hz)
    theta_d = torch.atan2(torch.sqrt(dx ** 2 + dy ** 2), dz)
    phi_d = torch.atan2(dy, dx)
    wiz = torch.cos(theta_d) * torch.cos(theta_h) - torch.sin(theta_d) * torch.cos(phi_d) * torch.sin(theta_h)
    rgb = brdf * torch.clip(wiz, 0, 1)
    return rgb




