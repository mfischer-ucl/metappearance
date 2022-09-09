import os
import json
import torch
import numpy as np
import torchvision.io as io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .metadataset import MetaDataset, MetaTask


class SVBRDFDataset(MetaDataset):

    def __init__(self, cfg, path, dataset_mode='train'):
        super(SVBRDFDataset, self).__init__()

        self.cfg = cfg
        self.path = path
        self.dataset_mode = dataset_mode
        self.append = cfg.data.use_pretrained_encoder

        img_dim = cfg.data.size
        self.resizeTransform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_dim[0], img_dim[1])),
            transforms.ToTensor()
        ])
        self.transform_hflip = transforms.RandomHorizontalFlip(p=1.0)
        self.transform_vflip = transforms.RandomVerticalFlip(p=1.0)

        # get data
        self.tasks = []
        self.create_tasks()

        if len(self.tasks) == 0:
            exit("Dataset Error. Could not find any SVBRDF data")

    # read the metadata.json file, which will be used to locate train and test sets
    def get_samples(self):
        with open(os.path.join(os.path.split(self.path)[0], 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        materials = metadata['materials']

        samples = []
        for material in materials:

            if material['split'] != self.dataset_mode:
                continue

            for entry in material['entries']:
                entry['id'] = material['material_id']
                entry['split'] = material['split']
                samples.append(entry)

        return samples

    def make_task(self, image, id):
        return SVBRDFTask(image=image,
                          id=id,
                          batchsize=self.cfg.train.bs,
                          w=self.cfg.model.w, z=self.cfg.model.z,
                          append=self.append,
                          device=self.cfg.device)

    def create_tasks(self):
        samples = self.get_samples()

        for idx, sample in enumerate(samples):
            if 'flash' in sample['name']: continue      # use henzler2021 only

            if self.cfg.data.use_only_first_n != -1:
                if idx == self.cfg.data.use_only_first_n:
                    break

            filepath = os.path.join(self.path, sample['id'], '{}{}'.format(sample['name'], sample['suffix']))

            image = io.read_image(filepath)
            image = self.resizeTransform(image)

            if self.dataset_mode == 'train':

                # make all augmented versions and create one task each
                img_h = self.transform_hflip(image)
                img_v = self.transform_vflip(image)
                img_hv = self.transform_hflip(img_v)

                if self.cfg.data.shuffle_c is True:
                    potential_cflips = torch.tensor([[0, 1, 2],     # rgb
                                                     [0, 2, 1],     # rbg
                                                     [1, 0, 2],     # grb
                                                     [1, 2, 0],     # ...
                                                     [2, 0, 1],
                                                     [2, 1, 0]])
                    for c in range(potential_cflips.shape[0]):
                        currflip = potential_cflips[c]
                        flip_id = '{}{}{}'.format(str(currflip[0]), str(currflip[1]), str(currflip[2]))
                        flipped_imgs = [x[currflip, ...] for x in [image, img_h, img_v, img_hv]]
                        for flipped_image, transf in zip(flipped_imgs, ['orig', 'flip_h', 'flip_v', 'flip_hv']):
                            task = self.make_task(image=flipped_image,
                                                  id='{}{}{}{}'.format(sample['name'],
                                                                       sample['suffix'],
                                                                       transf, flip_id))
                            self.tasks.append(task)
                else:
                    for transf_img, transf in zip([image, img_h, img_v, img_hv], ['orig', 'flip_h', 'flip_v', 'flip_hv']):
                        task = self.make_task(image=transf_img,
                                              id='{}{}{}'.format(sample['name'], sample['suffix'], transf))
                        self.tasks.append(task)
            else:
                task = self.make_task(image=image, id='{}{}'.format(sample['name'], sample['suffix']))
                self.tasks.append(task)


class SVBRDFTask(MetaTask):
    def __init__(self, image, id, batchsize, w=32, z=64, append=False, device='cuda'):
        super(SVBRDFTask, self).__init__()

        self.w = w
        self.z = z
        self.id = id
        self.bs = batchsize
        self.append = append
        self.__groundTruthImage = image.to(device)
        self.img_dim = [image.shape[1], image.shape[2]]

    def get_image(self):
        return self.__groundTruthImage.unsqueeze(0)

    def sample_noise(self):
        return torch.rand(self.bs, self.w, self.img_dim[0], self.img_dim[1], device=self.__groundTruthImage.device)

    # intra-task train and test is the same, because we're using random noise
    def get_trainbatch(self):
        return torch.cat([self.get_image() for _ in range(self.bs)], dim=0)

    def get_testbatch(self):
        return torch.cat([self.get_image() for _ in range(self.bs)], dim=0)

    def sample(self, mode='train'):
        img = self.get_trainbatch() if mode == 'train' else self.get_testbatch()
        noise = self.sample_noise()
        if self.append is True:
            # concat is necessary to comply with metappearance fw pass interface.
            # will be resolved in svbrdfmodel forward.
            noise = torch.cat([noise, img], dim=1)
        return noise, img

    def __len__(self):
        return 1
