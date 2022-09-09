import numpy as np


def save_model(model, h5, json=None):
    model.save_weights(h5)
    if json is None:
        json = h5.replace('.h5', '.json')
    with open(json, 'w') as f:
        f.write(model.to_json())


def normalize_phid(orig_phid):
    phid = orig_phid.copy()
    phid = np.where(phid < 0, phid + 2 * np.pi, phid)
    phid = np.where(phid >= 2 * np.pi, phid - 2 * np.pi, phid)
    return phid


def mask_from_array(arr):
    if len(arr.shape) > 1:
        mask = np.linalg.norm(arr, axis=1)
        mask[mask != 0] = 1
    else:
        mask = np.where(arr != 0, 1, 0)
    return mask
