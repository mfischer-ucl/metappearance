import math
import torch
import kornia
import torch.nn as nn
import torch.nn.functional as F


class Renderer(nn.Module):
    def __init__(self, fov, gamma, attenuation, type='ggx'):
        super(Renderer, self).__init__()

        self.type = type
        self.fov = fov
        self.gamma = gamma
        self.attenuation = attenuation

    def shading(self, diffuse, specular, roughness, normal, light_dir, view_dir):

        # shade
        if self.type == 'ggx':
            specular = getMicrofacetReflectance(light_dir, view_dir, specular,
                                                roughness, normal)
        elif self.type == 'phong':
            glossy_proxy = roughness
            specular = get_phong(light_dir, view_dir, specular, glossy_proxy,
                                 normal)
        elif self.type == 'phyiscal_phong':
            glossy_proxy = roughness
            specular = get_phong_phyiscal(light_dir, view_dir, specular,
                                          glossy_proxy, normal)
        elif self.type == 'blinn_phong':
            glossy_proxy = roughness
            specular = get_blinn_phong(light_dir, view_dir, specular,
                                       glossy_proxy, normal)
        else:
            raise NotImplementedError

        diffuse = get_diffuse(light_dir, diffuse, normal)
        shaded = math.pi * (diffuse + specular)

        return shaded

    def forward(self, brdf_maps, position, rotation_angle=None,
                light_pos_shift=None, use_newPosition=False, size=[192, 256], jitterPos=False, jitter=None):

        if use_newPosition is True and position is None:
            position = self.get_position(size=size)

        if jitterPos:
            assert jitter is not None
            # +1, links unten, -1 links oben
            # position is [1, 2, 192, 256] -> [b, xy, H, W] -> x-1 ist rechts mitte, y-1 ist oben mitte
            position[:, 0, :, :] += jitter[0]
            position[:, 1, :, :] += jitter[1]

        distance = 1 / math.tan(deg_to_rad(self.fov / 2))

        xy_zeros = torch.zeros_like(position[:, 0:2])
        z_ones = torch.ones_like(position[:, 0:1])

        image_pos = torch.cat([position, 0 * z_ones], dim=1)
        light_pos = torch.cat([xy_zeros, distance * z_ones], dim=1)
        view_pos = light_pos

        center_light_dir = torch.tensor(
            [0.0, 0.0, 1.0], device=position.device
        ).reshape(1, 3, 1, 1).expand(position.shape[0], 3, 1, 1)

        if light_pos_shift is not None:
            light_pos[:, :2] += light_pos_shift.unsqueeze(-1).unsqueeze(-1)
            # center_light_dir[:2] += light_pos_shift

        view_dir = F.normalize(view_pos - image_pos, p=2, dim=1)
        light_dir = F.normalize(light_pos - image_pos, p=2, dim=1)

        normal = brdf_maps['normal']

        if rotation_angle is not None:
            light_dir = rotate_by_rotation_angle(light_dir, rotation_angle)
            center_light_dir = rotate_by_rotation_angle(
                center_light_dir, rotation_angle
            )
            normal = rotate_by_rotation_angle(normal, rotation_angle)
            image_pos = rotate_by_rotation_angle(image_pos, rotation_angle)

        light_distance = torch.norm(light_pos - image_pos, dim=1, keepdim=True)

        radial_falloff = cos_angle(light_dir, center_light_dir)

        shaded = self.shading(brdf_maps['diffuse'],
                              brdf_maps['specular'],
                              brdf_maps['roughness'],
                              normal, light_dir, view_dir)

        if self.attenuation:
            shaded = shaded * self.light_decay(light_distance)
            shaded = shaded * self.radial_light_attenuation(radial_falloff)
            shaded = shaded * (2.4 ** 2)

        shaded = self.gamma_corr(shaded, self.gamma)
        shaded = torch.clamp(shaded, 0.0, 1.0)

        return shaded

    def radial_light_attenuation(self, radial_falloff):
        return radial_falloff ** 10.0

    def light_decay(self, light_distance):
        return 1.0 / ((light_distance ** 2) + 1e-4)

    def get_position(self, size):

        height, width = size
        aspect_ratio = width / height
        position = kornia.utils.create_meshgrid(height, width).permute(0, 3, 1, 2).to('cuda')
        position[:, 1] /= aspect_ratio
        position[:, 1] *= -1            # flip y axis as kornia uses different

        return position

    # gamma correction
    def gamma_corr(self, input, gamma=2.2):
        return input ** (1.0 / gamma)

    def log(self, tensor):
        eps = torch.tensor(0.01)
        return (torch.log(tensor + eps) - torch.log(eps)) \
               / (torch.log(1 + eps) - torch.log(eps))

    def height_to_normal(self, height):

        # calculate spatial gradients x,y from height field
        dx = (torch.roll(height, -1, dims=3) - torch.roll(height, 1, dims=3))
        dy = (torch.roll(height, 1, dims=2) - torch.roll(height, -1, dims=2))

        # retrieve normal based on spatial gradients
        e = 0.0001
        z = torch.ones_like(dx)
        nom = torch.cat([dx, dy, z], dim=1)
        denom = torch.sqrt(torch.sum(nom ** 2, dim=1, keepdim=True) + e)
        n = nom / denom

        return n

    def tonemap(self, input, percentile=99):
        bs = input.shape[0]
        n_elements = torch.numel(input[0])
        values, indices = torch.kthvalue(
            input.view(bs, -1), int(n_elements * (percentile / 100)), dim=1)
        exposure = 1 / (values + 0.00001)
        exposure = exposure.view(bs, 1, 1, 1)

        return exposure * input


def reflect(input, normal):
    # I - 2.0 * dot(N, I) * N.
    dot = torch.sum(input * normal, dim=1, keepdim=True)
    reflection = input - 2.0 * dot * normal
    return reflection


def glossy_to_glossy_proxy(glossy):
    return torch.sqrt(glossy) / 30.0


def glossy_proxy_to_glossy(glossy_proxy):
    return torch.exp(10 * glossy_proxy)
    return (30 * glossy_proxy) ** 2


def glossy_proxy_to_roughness(glossy_proxy):
    glossy = glossy_proxy_to_glossy(glossy_proxy)
    roughness = 6.5 * torch.pow(glossy + 29, -0.55)

    return roughness


def roughness_to_glossy_proxy(roughness):
    glossy = (6.5 / roughness) ** (100 / 55.0) - 29
    glossy_proxy = glossy_to_glossy_proxy(glossy)

    return glossy_proxy


def deg_to_rad(degree):
    return (degree / 180.0) * math.pi


def rad_to_deg(rad):
    return (rad / math.pi) * 180.0


##########

def get_diffuse(light_dir, diffuse, normal):
    k_d = torch.sum(light_dir * normal, dim=1, keepdim=True)
    k_d = torch.clamp(k_d, 0.0, 1.0)

    return (1 / math.pi) * diffuse * k_d


def sqr(a):
    return a ** 2


def ggx_microfacet(cos_half_angle, roughness):
    alpha2 = sqr(roughness)
    return sqr(alpha2) / (math.pi * sqr(
        sqr(cos_half_angle) * (sqr(alpha2) - 1.0) + 1.0) + 1e-6)


def cos_angle(a, b):
    angle = torch.sum(a * b, dim=1, keepdim=True)
    angle = torch.clamp(angle, 0.0, 1.0)

    return angle


def constant_fresnel(specular, cos_diff_angle):
    return specular


def schlick_fresnel(specular, cos_dif_angle):
    fresnel = specular + (1 - specular) * torch.pow(1.0 - cos_dif_angle, 5)
    return fresnel


def get_smiths_shadowing(roughness, cos_view_angle, cos_light_angle):
    alpha4 = sqr(sqr(roughness))
    G1o = 2.0 / (1.0 + torch.sqrt(
        1.0 + alpha4 * (1.0 - cos_view_angle * cos_view_angle) / (
                cos_view_angle * cos_view_angle + 1e-6)))
    G1i = 2.0 / (1.0 + torch.sqrt(
        1.0 + alpha4 * (1.0 - cos_light_angle * cos_light_angle) / (
                cos_light_angle * cos_light_angle + + 1e-6)))

    return G1o * G1i


def getMicrofacetReflectance(light_dir, view_dir, specular, roughness, normal):
    half_direction = F.normalize(0.5 * (light_dir + view_dir), p=2, dim=1)
    cos_half_angle = cos_angle(half_direction, normal)
    cos_light_angle = cos_angle(light_dir, normal)

    cos_view_angle = cos_angle(view_dir, normal)
    cos_diff_angle = cos_angle(half_direction, light_dir)

    distribution = ggx_microfacet(cos_half_angle, roughness)
    # fresnel = constant_fresnel(specular, cos_diff_angle)
    fresnel = schlick_fresnel(specular, cos_diff_angle)
    geoemtry = get_smiths_shadowing(roughness, cos_view_angle, cos_light_angle)

    reflectance = (distribution * geoemtry * fresnel)
    reflectance = reflectance / (4.0 * cos_light_angle + 1e-6)

    return reflectance


def get_phong_phyiscal(light_dir, view_dir, specular, glossy_proxy, normal):
    glossy = glossy_proxy_to_glossy(glossy_proxy)
    specular = light_dir, view_dir, specular, glossy_proxy, normal
    return ((glossy + 2) / (2 * math.pi)) * specular * math.pi


def get_phong(light_dir, view_dir, specular, glossy_proxy, normal):
    reflect_dir = reflect(light_dir, normal)
    k_s = cos_angle(view_dir, reflect_dir)
    glossy = glossy_proxy_to_glossy(glossy_proxy)
    k_s = torch.pow(k_s + 1e-8, glossy)

    return (1 / math.pi) * specular * k_s


def get_blinn_phong(light_dir, view_dir, specular, glossy_proxy, normal):
    half_direction = F.normalize(0.5 * (light_dir + view_dir), p=2, dim=1)
    k_s = cos_angle(normal, half_direction)
    glossy = glossy_proxy_to_glossy(glossy_proxy)
    k_s = torch.pow(k_s + 1e-8, glossy)

    return (1 / math.pi) * specular * k_s


def rotate(array, angle, axis=0):
    bs, c, h, w = array.size()

    device = 'cuda' if array.is_cuda else 'cpu'
    rotation_matrix = torch.zeros((bs, 3, 3), device=device)

    if axis == 0:
        rotation_matrix[:, 0, 0] = 1.0
        rotation_matrix[:, 1, 1] += torch.cos(angle)
        rotation_matrix[:, 1, 2] += -torch.sin(angle)
        rotation_matrix[:, 2, 1] += torch.sin(angle)
        rotation_matrix[:, 2, 2] += torch.cos(angle)
    elif axis == 1:
        rotation_matrix[:, 0, 0] += torch.cos(angle.squeeze())
        rotation_matrix[:, 0, 2] += torch.sin(angle)
        rotation_matrix[:, 1, 1] = 1.0
        rotation_matrix[:, 2, 0] += -torch.sin(angle)
        rotation_matrix[:, 2, 2] += torch.cos(angle)
    else:
        raise NotImplementedError

    rotation_matrix = rotation_matrix.expand(bs, 3, 3)
    array_rotated = torch.bmm(rotation_matrix, array.view(bs, 3, -1))
    array_rotated = array_rotated.view(bs, 3, h, w)

    return array_rotated


def rotate_by_rotation_angle(tensor, rotation_angle):

    tensor = rotate(tensor, (math.pi / 8.0) * rotation_angle[:, 0], axis=1)
    tensor = rotate(tensor, (math.pi / 8.0) * rotation_angle[:, 1], axis=0)

    return tensor


