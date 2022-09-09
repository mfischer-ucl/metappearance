import os
import cv2
import torch
import numpy as np
from models import losses


def resolve_data(mode, basepath, appl_subdict, skip_train=False):
    trainpath = os.path.join(basepath, appl_subdict.data.trainpath)
    testpath = os.path.join(basepath, appl_subdict.data.testpath)
    if mode == 'texture':
        from datasets.texturedataset import TextureDataset
        trainDistr = TextureDataset(appl_subdict, trainpath, dataset_mode='train') if skip_train is False else None
        testDistr = TextureDataset(appl_subdict, testpath, dataset_mode='test')
    if mode == 'brdf':
        from datasets.brdfdataset import BRDFDataset
        trainDistr = BRDFDataset(appl_subdict, trainpath) if skip_train is False else None
        testDistr = BRDFDataset(appl_subdict, testpath)
    if mode == 'svbrdf':
        from datasets.svbrdfdataset import SVBRDFDataset
        trainDistr = SVBRDFDataset(appl_subdict, trainpath, dataset_mode='train') if skip_train is False else None
        testDistr = SVBRDFDataset(appl_subdict, testpath, dataset_mode='test')

    return trainDistr, testDistr


def resolve_model(mode, appl_subdict, verbose):
    if mode == 'texture':
        from models.texturemodel import Model
    if mode == 'brdf':
        from models.brdfmodel import Model
    if mode == 'svbrdf':
        from models.svbrdfmodel import Model

    model = Model(appl_subdict)
    if verbose:
        from torchinfo import summary
        summary(model)
    return model


def resolve_loss(mode, appl_subdict):
    if mode == 'texture':
        return losses.TextureLoss()
    if mode == 'brdf':
        return losses.BRDFLoss()
    if mode == 'svbrdf':
        return losses.SVBRDFLoss(appl_subdict)


def zero_gradients(params):
    # clear all previous gradients
    for p in params:
        if p.is_leaf:
            if p.grad is not None:
                p.grad.zero_()


def load_image(path, return_torch=False):
    if not os.path.isfile(path):
        raise FileNotFoundError(str(path))

    image = cv2.imread(path)

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError

    if image.ndim == 2:
        image = image[..., np.newaxis]

    if image is None:
        raise FileNotFoundError

    if image.dtype == np.uint8 or image.dtype == np.uint16:
        image = convert_to_float(image)

    if return_torch:
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    return image


def convert_to_float(array):
    max_value = np.iinfo(array.dtype).max
    array[array > max_value] = max_value

    if type(array).__module__ == 'numpy':
        return array.astype(np.float32) / max_value

    elif type(array).__module__ == 'torch':
        return array.float() / max_value
    else:
        raise NotImplementedError
