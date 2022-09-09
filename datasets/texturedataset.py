import os
import torch
import numpy as np
from utils.utils import load_image
import torchvision.transforms as transforms
from .metadataset import MetaDataset, MetaTask


class TextureDataset(MetaDataset):

    def __init__(self, cfg, path, dataset_mode='train'):
        super(TextureDataset, self).__init__()

        self.cfg = cfg
        self.path = path
        self.dataset_mode = dataset_mode

        img_dim = cfg.data.size
        self.resizeTransform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_dim[0], img_dim[1])),
            transforms.ToTensor()
        ])

        self.tasks = []
        self.create_tasks()

        if len(self.tasks) == 0:
            exit("Dataset Error. Could not find any texture data.")

    def create_tasks(self):

        for idx, imgname in enumerate(os.listdir(self.path)):

            if self.cfg.data.use_only_first_n != -1:
                if idx == self.cfg.data.use_only_first_n:
                    break

            filepath = os.path.join(self.path, imgname)

            image = load_image(filepath, return_torch=True).to(self.cfg.device)
            image = self.resizeTransform(image)

            if self.dataset_mode == 'train':

                # augment: for each image, create vertically and horizontally flipped versions
                img_h = transforms.RandomHorizontalFlip(p=1.0)(image)
                img_v = transforms.RandomVerticalFlip(p=1.0)(image)
                img_hv = transforms.RandomVerticalFlip(p=1.0)(img_h)

                for img, descr in zip([image, img_h, img_v, img_hv], ['orig', 'flip_h', 'flip_v', 'flip_hv']):
                    task = TextureTask(image=img,
                                       batchsize=self.cfg.train.bs,
                                       id=imgname+descr,
                                       img_dim=self.cfg.data.size,
                                       w=self.cfg.model.w, z=self.cfg.model.z,
                                       device=self.cfg.device)
                    self.tasks.append(task)
            else:
                task = TextureTask(image=image,
                                   batchsize=self.cfg.train.bs,
                                   id=imgname,
                                   img_dim=self.cfg.data.size,
                                   w=self.cfg.model.w, z=self.cfg.model.z,
                                   device=self.cfg.device)
                self.tasks.append(task)


class TextureTask(MetaTask):
    def __init__(self, image, batchsize, id, img_dim, w=32, z=64, device='cuda'):
        super(MetaTask, self).__init__()

        self.w = w
        self.z = z
        self.id = id
        self.bs = batchsize
        self.img_dim = img_dim

        self.__groundTruthImage = image.to(device)

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
        return noise, img

    def __len__(self):
        return 1



