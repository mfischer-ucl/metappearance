import os
import sys
import torch
import numpy as np
import pandas as pd
from .metadataset import MetaDataset, MetaTask

sys.path.append("..")  # add higher directory to python modules path.
from utils.brdf import fastmerl, common, coords


class BRDFDataset(MetaDataset):

    def __init__(self, cfg, path):
        super(BRDFDataset, self).__init__()
        self.cfg = cfg
        self.path = path

        self.tasks = []
        self.create_tasks()

        if len(self.tasks) == 0:
            exit("Dataset Error. Could not find any BRDF data.")

    def create_tasks(self):
        # read all MERL files from folder
        for idx, merlfile in enumerate(os.listdir(self.path)):
            if self.cfg.data.use_only_first_n != -1:
                if idx == self.cfg.data.use_only_first_n:
                    break
            if not merlfile.endswith('binary'): continue

            task = BRDFTask(path=os.path.join(self.path, merlfile),
                            batchsize=self.cfg.train.bs,
                            id=merlfile.replace('.binary', ''),
                            device=self.cfg.device)

            self.tasks.append(task)
            # print("Read and loaded {}".format(merlfile))


class BRDFTask(MetaTask):
    def __init__(self, path, batchsize, id, device='cuda'):
        super(MetaTask, self).__init__()

        yvars = ['brdf_r', 'brdf_g', 'brdf_b']
        xvars = ['hx', 'hy', 'hz', 'dx', 'dy', 'dz']

        self.id = id
        self.bs = batchsize

        # create BRDF from merlpath
        self.BRDF = fastmerl.Merl(path)

        self.reflectance_train = generate_nn_datasets(self.BRDF, xvars=xvars, yvars=yvars, nsamples=800000, pct=0.8)
        self.reflectance_test = generate_nn_datasets(self.BRDF, xvars=xvars, yvars=yvars, nsamples=800000, pct=0.2)

        self.train_samples = torch.tensor(self.reflectance_train[xvars].values, dtype=torch.float32, device=device)
        self.train_gt = torch.tensor(self.reflectance_train[yvars].values, dtype=torch.float32, device=device)

        self.test_samples = torch.tensor(self.reflectance_test[xvars].values, dtype=torch.float32, device=device)
        self.test_gt = torch.tensor(self.reflectance_test[yvars].values, dtype=torch.float32, device=device)

    def __len__(self):
        return self.train_samples.shape[0]

    def get_trainbatch(self, idx=None):
        if idx is None:
            idx = np.random.randint(0, len(self.train_samples) - self.bs)
        return self.train_samples[idx:idx + self.bs, :], self.train_gt[idx:idx + self.bs, :]

    def get_testbatch(self, idx=None):
        if idx is None:
            idx = np.random.randint(0, len(self.test_samples) - self.bs)
        return self.test_samples[idx:idx + self.bs, :], self.test_gt[idx:idx + self.bs, :]

    def shuffle(self):
        r = torch.randperm(self.train_samples.shape[0])
        self.train_samples = self.train_samples[r, :]
        self.train_gt = self.train_gt[r, :]

    def sample(self, mode='train'):
        # get a random batch from either test or trainset
        if mode == 'train':
            idx = np.random.randint(0, len(self.train_samples) - self.bs)
            return self.get_trainbatch(idx)
        elif mode == 'test':
            idx = np.random.randint(0, len(self.test_samples) - self.bs)
            return self.get_testbatch(idx)


def generate_nn_datasets(brdf, xvars, yvars, nsamples=800000, pct=0.8):
    rangles = np.random.uniform([0, 0, 0], [np.pi / 2., np.pi / 2., 2 * np.pi], [int(nsamples * pct), 3]).T
    rangles[2] = common.normalize_phid(rangles[2])

    rvectors = coords.rangles_to_rvectors(*rangles)
    brdf_vals = brdf_values(rvectors, brdf=brdf)

    df = pd.DataFrame(np.concatenate([rvectors.T, brdf_vals], axis=1), columns=[*xvars, *yvars])
    df = df[(df.T != 0).any()]
    df = df.drop(df[df['brdf_r'] < 0].index)
    return df


def brdf_values(rvectors, brdf=None):
    if brdf is not None:
        rangles = coords.rvectors_to_rangles(*rvectors)
        brdf_arr = brdf.eval_interp(*rangles).T
    else:
        raise RuntimeError("Provided BRDF is None.")
    brdf_arr *= common.mask_from_array(rvectors.T).reshape(-1, 1)
    return brdf_arr
