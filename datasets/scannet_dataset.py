import os
import glob
import json

import h5py
import numpy as np
from PIL import Image
from termcolor import colored, cprint

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from datasets.base_dataset import BaseDataset

from utils.util_3d import read_sdf
from utils import binvox_rw

class ScanNetImgDataset(BaseDataset):
    def initialize(self, opt, phase='train', cat='all'):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.phase = phase

        with open(f'/home/xiong/PT43D/dataset_info_files/info-shapenet.json') as f:
            self.info = json.load(f)

        self.code_dir = f'/mnt/raid/xiong/ShapeNet/extracted_code/pvqvae-snet-all-T0.2'
        assert os.path.exists(self.code_dir), f'{self.code_dir} should exist.'

        self.cat_to_id = self.info['cats']
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}

        if cat == 'all':
            cats = self.info['all_cats']
        else:
            cats = [cat]
        self.img_list = []
        self.model_list = []
        self.cats_list = []
        self.sdf_list = []
        self.vox_list = []
        self.model_id_list = []
        if phase == 'test':
            file_name = 'scannet_val'
        else:
            file_name = 'scannet_train'
        for c in cats:
            synset = self.info['cats'][c]
            with open(f'/home/xiong/PT43D/dataset_info_files/ScanNet_filelists/{synset}_{phase}.lst') as f:
                img_list_s = []
                model_list_s = []
                sdf_list_s = []
                vox_list_s = []
                model_id_list_s = []
                for l in f.readlines():
                    img_id = l.rstrip('\n')

                    # for rendered img
                    render_img_dir = f'/mnt/raid/xiong/{file_name}/{synset}/{img_id}.pt'
                    if not os.path.exists(render_img_dir):
                        continue
                    # for sdf
                    with open(f'/mnt/raid/xiong/{file_name}/{synset}/{img_id}.txt') as f2:
                        model_id = f2.readlines()[0].rstrip('\n')
                    sdf_path = f'/mnt/raid/xiong/ShapeNet/SDF_v1/resolution_64/{synset}/{model_id}/ori_sample_grid.h5'
                    if not os.path.exists(sdf_path):
                        continue
                    sdf_list_s.append(sdf_path)
                    render_img_list = [render_img_dir]
                    model_id_list_s.append(model_id)

                    # for img
                    img_list_s.append(render_img_list)

                    # for code
                    code_path = f'{self.code_dir}/{synset}/{model_id}'
                    model_list_s.append(code_path)

                    # for vox
                    vox_path = os.path.join('/mnt/raid/xiong', 'ShapeNet', 'ShapeNetVox32', synset, model_id,
                                            'model.binvox')
                    vox_list_s.append(vox_path)

                self.img_list += img_list_s
                self.model_list += model_list_s
                self.sdf_list += sdf_list_s
                self.vox_list += vox_list_s
                self.model_id_list += model_id_list_s
                self.cats_list += [synset] * len(img_list_s)
                # print('[*] %d samples for %s (%s).' % (len(img_list_s), shapenet_dict['id_to_cat'][synset], synset))
                print('[*] %d samples for %s (%s).' % (len(model_list_s), self.id_to_cat[synset], synset))

        np.random.default_rng(seed=0).shuffle(self.img_list)
        np.random.default_rng(seed=0).shuffle(self.model_list)
        np.random.default_rng(seed=0).shuffle(self.sdf_list)
        np.random.default_rng(seed=0).shuffle(self.vox_list)
        np.random.default_rng(seed=0).shuffle(self.cats_list)
        np.random.default_rng(seed=0).shuffle(self.model_id_list)

        # need to check the seed for reproducibility
        self.img_list = self.img_list[:self.max_dataset_size]
        self.model_list = self.model_list[:self.max_dataset_size]
        self.sdf_list = self.sdf_list[:self.max_dataset_size]
        self.vox_list = self.vox_list[:self.max_dataset_size]
        self.cats_list = self.cats_list[:self.max_dataset_size]
        self.model_id_list = self.model_id_list[:self.max_dataset_size]
        cprint('[*] %d img_list loaded.' % (len(self.img_list)), 'yellow')
        cprint('[*] %d code loaded.' % (len(self.model_list)), 'yellow')

        assert len(self.img_list) == len(self.model_list) == len(self.vox_list) == len(self.cats_list) == len(
            self.sdf_list) == len(self.model_id_list)

        self.N = len(self.img_list)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.resize = transforms.Resize((256, 256))

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        if phase == 'train':
            self.transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.RandomAffine(0, scale=(0.7, 1.25), interpolation=InterpolationMode.BILINEAR),
                transforms.Normalize(mean, std),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Normalize(mean, std),
                # transforms.Resize((256, 256)),
            ])

        self.transforms_bg = transforms.Compose([
            transforms.RandomCrop(256, pad_if_needed=True, padding_mode='padding_mode'),
            transforms.Normalize(mean, std),
        ])

    def process_img(self, img):
        img_t = self.to_tensor(img)

        _, oh, ow = img_t.shape

        ls = max(oh, ow)

        pad_h1, pad_h2 = (ls - oh) // 2, (ls - oh) - (ls - oh) // 2
        pad_w1, pad_w2 = (ls - ow) // 2, (ls - ow) - (ls - ow) // 2

        img_t = F.pad(img_t[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2), mode='constant', value=0)[0]

        if self.phase == 'train':
            img_fg_mask = (img_t != 0.).float()
            # jitter color first
            img_t = self.transforms_color(img_t)
            img_t_with_mask = torch.cat([img_t, img_fg_mask], dim=0)
            img_t_with_mask = self.transforms(img_t_with_mask)
            img_t, img_fg_mask = img_t_with_mask[:3], img_t_with_mask[3:]
            img_fg_mask = self.resize(img_fg_mask)
            img_t = self.normalize(img_t)
            img_t = self.resize(img_t)
        else:
            img_t = self.transforms(img_t)

        return img_t

    def process_img_t(self, img_t):
        if self.phase == 'train':  # self.opt.pix3d_mode == 'noBG':
            _, oh, ow = img_t.shape

            ls = max(oh, ow)

            pad_h1, pad_h2 = (ls - oh) // 2, (ls - oh) - (ls - oh) // 2
            pad_w1, pad_w2 = (ls - ow) // 2, (ls - ow) - (ls - ow) // 2
            img_t = F.pad(img_t[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2), mode='constant', value=0)
            img_t = self.transforms(img_t[0])
        else:
            img_t = self.transforms(img_t)
        return img_t

    def read_vox(self, vox_path):
        with open(vox_path, 'rb') as f:
            vox = binvox_rw.read_as_3d_array(f).data.astype(np.uint8)

        vox = torch.from_numpy(vox).float()  # [None, ...].to(device)
        return vox

    def __getitem__(self, index):
        model_id = self.model_id_list[index]
        imgs = []
        img_paths = []
        imgs_all_view = self.img_list[index]
        # allow replacement. cause in test time, we might only see images from one view
        nimgs = 1
        sample_ixs = np.random.choice(len(imgs_all_view), nimgs)
        for ix in sample_ixs:
            p = imgs_all_view[ix]
            im = torch.load(p)
            im = self.process_img_t(im)
            imgs.append(im)
            img_paths.append(p)

        imgs = torch.stack(imgs)
        img = imgs[0]
        img_path = img_paths[0]

        if self.phase == 'train':
            sample_model_id = model_id
        else:
            sample_model_id = model_id  # no mapping
        mapped_index = self.model_id_list.index(sample_model_id)
        synset = self.cats_list[mapped_index]
        model = self.model_list[mapped_index]
        gt_vox_path = self.vox_list[mapped_index]

        sdf_p = self.sdf_list[mapped_index]
        code_p = f'{model}/code.npy'
        codeix_p = f'{model}/codeix.npy'

        sdf = read_sdf(sdf_p).squeeze(0)

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        code = torch.from_numpy(np.load(code_p))
        codeix = torch.from_numpy(np.load(codeix_p))
        gt_vox = self.read_vox(gt_vox_path)

        ret = {

            'sdf': sdf,
            'z_q': code,
            'idx': codeix,
            'img': img,
            'imgs': imgs,
            'cat_id': synset,
            'cat_str': self.id_to_cat[synset],
            'path': model,
            'img_path': img_path,
            'img_paths': img_paths,
            'gt_vox': gt_vox,
            'gt_vox_path': gt_vox_path,
            'model_id': model_id,
            'visible_points_path': img_path.split('.')[0] + '.npy'
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'ScanNetImageDataset'