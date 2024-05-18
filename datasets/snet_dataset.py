"""
    adopted from: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/shapenet.py
"""
import os.path
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

import random


def get_code_setting(opt):
    code_setting = f'{opt.vq_model}-{opt.vq_dset}-{opt.vq_cat}-T{opt.trunc_thres}'
    if opt.vq_note != 'default':
        code_setting = f'{code_setting}-{opt.vq_note}'
    return code_setting


# from https://github.com/laughtervv/DISN/blob/master/preprocessing/info.json
class ShapeNetDataset(BaseDataset):

    def initialize(self, opt, phase='train', cat='all'):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size

        with open(f'/home/xiong/PT43D/dataset_info_files/info-shapenet.json') as f:
            self.info = json.load(f)

        self.cat_to_id = self.info['cats']
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}

        if cat == 'all':
            all_cats = self.info['all_cats']
        else:
            all_cats = [cat]
        self.phase = phase
        self.model_list = []
        self.cats_list = []
        for c in all_cats:
            synset = self.info['cats'][c]
            with open(f'/home/xiong/PT43D/dataset_info_files/ShapeNet_filelists/{synset}_{phase}.lst') as f:
                model_list_s = []
                for l in f.readlines():
                    model_id = l.rstrip('\n')
                    path = f'/mnt/raid/xiong/ShapeNet/SDF_v1/resolution_64/{synset}/{model_id}/ori_sample_grid.h5'
                    model_list_s.append(path)

                self.model_list += model_list_s
                self.cats_list += [synset] * len(model_list_s)
                print('[*] %d samples for %s (%s).' % (len(model_list_s), self.id_to_cat[synset], synset))

        np.random.default_rng(seed=0).shuffle(self.model_list)
        np.random.default_rng(seed=0).shuffle(self.cats_list)


        # cprint('[*] (SDFDataset) there are %d categories.' % (len(all_catids)), 'yellow')

        # need to check the seed for reproducibility
        self.model_list = self.model_list[:self.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.model_list)), 'yellow')

        self.N = len(self.model_list)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __getitem__(self, index):
        # model_id = self.model_list[index]
        synset = self.cats_list[index]
        # sdf_h5_file = osp.join(self._data_dir, 'SDF_v1', synset, model_id, 'ori_sample_grid.h5')
        sdf_h5_file = self.model_list[index]

        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, 64, 64, 64)
        # print(sdf.shape)
        # sdf = sdf[:, :64, :64, :64]

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        ret = {
            'sdf': sdf,
            'cat_id': synset,
            'cat_str': self.id_to_cat[synset],
            'path': sdf_h5_file,
            # 'tsdf': tsdf,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'SDFDataset'


class ShapeNetCodeDataset(BaseDataset):

    # def initialize(self, opt, phase='train', cat='chair'):
    def initialize(self, opt, phase='train', cat='all'):
        self.opt = opt
        self.ratio = opt.ratio
        self.max_dataset_size = opt.max_dataset_size
        self.phase = phase

        with open(f'/home/xiong/PT43D/dataset_info_files/info-shapenet.json') as f:
            self.info = json.load(f)

        self.cat_to_id = self.info['cats']
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}

        code_setting = get_code_setting(opt)
        self.code_dir = f'/mnt/raid/xiong/ShapeNet/extracted_code/{code_setting}'
        assert os.path.exists(self.code_dir), f'{self.code_dir} should exist.'

        if cat == 'all':
            all_cats = self.info['all_cats']
        else:
            all_cats = [cat]
        # all_cats = ['chair', 'table']
        self.model_list = []
        self.model_id_list = []
        self.img_list = []
        self.cats_list = []
        self.mapping_list = []
        self.img2shapes_mapping = {}
        for c in all_cats:
            synset = self.info['cats'][c]
            with open(f'/home/xiong/PT43D/dataset_info_files/image2shapes_from_geo_mapping_{c}.json') as f:
                self.img2shapes_mapping.update(json.load(f))
            with open(f'/home/xiong/PT43D/dataset_info_files/ShapeNet_filelists/{synset}_{phase}.lst') as f:
                model_list_s = []
                model_id_list_s = []
                img_list_s = []
                mapping_list_s = []
                for l in f.readlines():
                    model_id = l.rstrip('\n')

                    # for rendered img
                    render_img_dir = os.path.join('/mnt/raid/xiong', 'ShapeNet', 'Rendering_6', synset, model_id)
                    if not os.path.exists(render_img_dir):
                        continue
                    render_img_list = [os.path.join(render_img_dir, f) for f in os.listdir(render_img_dir) if
                                       'label.png' not in f and 'label.npy' not in f and '.txt' not in f]

                    model_id_list_s.append(model_id)

                    # for img
                    img_list_s.append(render_img_list)

                    path = f'{self.code_dir}/{synset}/{model_id}'
                    model_list_s.append(path)

                    mapping_list = [os.path.join(render_img_dir, f) for f in os.listdir(render_img_dir) if '.txt' in f]
                    mapping_list_s.append(mapping_list)

                nimgs_img_list_s = len(model_list_s)
                nimgs_to_take = int(nimgs_img_list_s * self.ratio)

                # self.model_list += model_list_s
                # self.cats_list += [synset] * len(model_list_s)
                self.img_list += img_list_s[:nimgs_to_take]
                self.mapping_list += mapping_list_s[:nimgs_to_take]
                self.model_id_list += model_id_list_s[:nimgs_to_take]
                self.model_list += model_list_s[:nimgs_to_take]
                self.cats_list += [synset] * len(model_list_s[:nimgs_to_take])
                print('[*] %d samples for %s (%s).' % (len(model_list_s), self.id_to_cat[synset], synset))

        np.random.default_rng(seed=0).shuffle(self.model_list)
        np.random.default_rng(seed=0).shuffle(self.cats_list)
        np.random.default_rng(seed=0).shuffle(self.img_list)
        np.random.default_rng(seed=0).shuffle(self.model_id_list)
        np.random.default_rng(seed=0).shuffle(self.mapping_list)

        # cprint('[*] (SDFDataset) there are %d categories.' % (len(all_catids)), 'yellow')

        # need to check the seed for reproducibility
        self.model_list = self.model_list[:self.max_dataset_size]
        self.model_id_list = self.model_id_list[:self.max_dataset_size]
        self.cats_list = self.cats_list[:self.max_dataset_size]
        self.img_list = self.img_list[:self.max_dataset_size]
        self.mapping_list = self.mapping_list[:self.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.model_list)), 'yellow')

        assert (len(self.img_list) == len(self.model_list) == len(self.cats_list) == len(self.model_id_list))# == len(self.mapping_list)

        self.N = len(self.model_list)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.resize = transforms.Resize((256, 256))

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        if phase == 'train':
            self.transforms_color = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
            self.transforms = transforms.Compose([
                transforms.RandomAffine(0, scale=(0.7, 1.25), interpolation=InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Normalize(mean, std),
                transforms.Resize((256, 256)),
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

    def __getitem__(self, index):
        model_id = self.model_id_list[index]
        imgs = []
        img_paths = []
        imgs_all_view = self.img_list[index]
        mappings_all_view = self.mapping_list[index]
        # allow replacement. cause in test time, we might only see images from one view
        nimgs = 1
        sample_ixs = np.random.choice(len(imgs_all_view), nimgs)
        for ix in sample_ixs:
            p = imgs_all_view[ix]
            im = Image.open(p).convert('RGB')
            im = self.process_img(im)
            imgs.append(im)
            img_paths.append(p)

        imgs = torch.stack(imgs)
        img = imgs[0]
        img_path = img_paths[0]

        if self.phase == 'train':
            current_mapping = [model_id]
            with open(mappings_all_view[sample_ixs[0]]) as f:
                for l in f.readlines():
                    mapping_model_id = l.rstrip('\n')
                    current_mapping.append(mapping_model_id)
            sample_model_id = current_mapping[random.randint(0, len(current_mapping) - 1)]  # accurate mapping
            # sample_model_id = model_id  # no mapping
            # sample_model_id = self.img2shapes_mapping[model_id][
            #     random.randint(0, len(self.img2shapes_mapping[model_id]) - 1)] # coarse mapping
            # if sample_model_id not in self.model_id_list:
            #     sample_model_id = model_id
        else:
            sample_model_id = model_id  # no mapping
        mapped_index = self.model_id_list.index(sample_model_id)
        synset = self.cats_list[mapped_index]
        model = self.model_list[mapped_index]

        sdf_p = f'{model}/x.npy'
        code_p = f'{model}/code.npy'
        codeix_p = f'{model}/codeix.npy'

        origin_model = self.model_list[index]
        origin_sdf_p = f'{origin_model}/x.npy'
        origin_code_p = f'{origin_model}/code.npy'
        origin_codeix_p = f'{origin_model}/codeix.npy'


        sdf = torch.from_numpy(np.load(sdf_p))
        code = torch.from_numpy(np.load(code_p))
        codeix = torch.from_numpy(np.load(codeix_p))

        origin_sdf = torch.from_numpy(np.load(origin_sdf_p))
        origin_code = torch.from_numpy(np.load(origin_code_p))
        origin_codeix = torch.from_numpy(np.load(origin_codeix_p))

        ret = {
            'sdf': sdf,
            'z_q': code,
            'idx': codeix,
            'origin_sdf': origin_sdf,
            'origin_z_q': origin_code,
            'origin_idx': origin_codeix,

            'cat_id': synset,
            'cat_str': self.id_to_cat[synset],
            'path': model,

            'img': img,
            'imgs': imgs,
            'img_path': img_path,
            'img_paths': img_paths,
            'model_id': model_id,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'ShapeNetCodeDataset'


class ShapeNetImgDataset(BaseDataset):

    # def initialize(self, opt, phase='train', cat='chair'):
    def initialize(self, opt, phase='train', cat='all'):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.snet_mode = opt.snet_mode
        self.phase = phase

        with open(f'/home/xiong/PT43D/dataset_info_files/info-shapenet.json') as f:
            self.info = json.load(f)

        # code_setting = f'{opt.vq_model}-{opt.vq_dset}-{opt.vq_cat}-T{opt.trunc_thres}'
        code_setting = get_code_setting(opt)
        self.code_dir = f'/mnt/raid/xiong/ShapeNet/extracted_code/{code_setting}'
        assert os.path.exists(self.code_dir), f'{self.code_dir} should exist.'

        self.cat_to_id = self.info['cats']
        self.id_to_cat = {v: k for k, v in self.cat_to_id.items()}

        if cat == 'all':
            cats = self.info['all_cats']
        else:
            cats = [cat]
        # cats = ['chair', 'table']
        self.img_list = []
        self.model_list = []
        self.cats_list = []
        self.sdf_list = []
        self.vox_list = []
        self.model_id_list = []
        self.mapping_list = []
        for c in cats:
            synset = self.info['cats'][c]
            with open(f'/home/xiong/PT43D/dataset_info_files/ShapeNet_filelists/{synset}_{phase}.lst') as f:
                img_list_s = []
                model_list_s = []
                sdf_list_s = []
                vox_list_s = []
                model_id_list_s = []
                mapping_list_s = []
                for l in f.readlines():
                    model_id = l.rstrip('\n')

                    # for rendered img
                    render_img_dir = os.path.join('/mnt/raid/xiong', 'ShapeNet', 'Rendering_6', synset, model_id)
                    if not os.path.exists(render_img_dir):
                        continue
                    render_img_list = [os.path.join(render_img_dir, f) for f in os.listdir(render_img_dir) if
                                       'label.png' not in f and 'label.npy' not in f and '.txt' not in f]

                    model_id_list_s.append(model_id)

                    # for img
                    img_list_s.append(render_img_list)

                    # for code
                    code_path = f'{self.code_dir}/{synset}/{model_id}'
                    model_list_s.append(code_path)

                    # for sdf
                    sdf_path = f'/mnt/raid/xiong/ShapeNet/SDF_v1/resolution_64/{synset}/{model_id}/ori_sample_grid.h5'
                    sdf_list_s.append(sdf_path)

                    # for vox
                    vox_path = os.path.join('/mnt/raid/xiong', 'ShapeNet', 'ShapeNetVox32', synset, model_id,
                                            'model.binvox')
                    vox_list_s.append(vox_path)

                    mapping_list = [os.path.join(render_img_dir, f) for f in os.listdir(render_img_dir) if '.txt' in f]
                    mapping_list_s.append(mapping_list)

                self.img_list += img_list_s
                self.model_list += model_list_s
                self.sdf_list += sdf_list_s
                self.vox_list += vox_list_s
                self.model_id_list += model_id_list_s
                self.cats_list += [synset] * len(img_list_s)
                self.mapping_list += mapping_list_s
                # print('[*] %d samples for %s (%s).' % (len(img_list_s), shapenet_dict['id_to_cat'][synset], synset))
                print('[*] %d samples for %s (%s).' % (len(model_list_s), self.id_to_cat[synset], synset))

        np.random.default_rng(seed=0).shuffle(self.img_list)
        np.random.default_rng(seed=0).shuffle(self.model_list)
        np.random.default_rng(seed=0).shuffle(self.sdf_list)
        np.random.default_rng(seed=0).shuffle(self.vox_list)
        np.random.default_rng(seed=0).shuffle(self.cats_list)
        np.random.default_rng(seed=0).shuffle(self.model_id_list)
        np.random.default_rng(seed=0).shuffle(self.mapping_list)

        # cprint('[*] (SDFDataset) there are %d categories.' % (len(all_catids)), 'yellow')

        # need to check the seed for reproducibility
        self.img_list = self.img_list[:self.max_dataset_size]
        self.model_list = self.model_list[:self.max_dataset_size]
        self.sdf_list = self.sdf_list[:self.max_dataset_size]
        self.vox_list = self.vox_list[:self.max_dataset_size]
        self.cats_list = self.cats_list[:self.max_dataset_size]
        self.model_id_list = self.model_id_list[:self.max_dataset_size]
        self.mapping_list = self.mapping_list[:self.max_dataset_size]
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
            self.transforms_color = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
            self.transforms = transforms.Compose([
                transforms.RandomAffine(0, scale=(0.7, 1.25), interpolation=InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Normalize(mean, std),
                transforms.Resize((256, 256)),
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
        mappings_all_view = self.mapping_list[index]
        # allow replacement. cause in test time, we might only see images from one view
        nimgs = 1
        sample_ixs = np.random.choice(len(imgs_all_view), nimgs)
        for ix in sample_ixs:
            p = imgs_all_view[ix]
            im = Image.open(p).convert('RGB')
            im = self.process_img(im)
            imgs.append(im)
            img_paths.append(p)

        imgs = torch.stack(imgs)
        img = imgs[0]
        img_path = img_paths[0]

        if self.phase == 'train':
            current_mapping = [model_id]
            with open(mappings_all_view[sample_ixs[0]]) as f:
                for l in f.readlines():
                    mapping_model_id = l.rstrip('\n')
                    current_mapping.append(mapping_model_id)
            sample_model_id = current_mapping[random.randint(0, len(current_mapping) - 1)]  # accurate mapping
            # sample_model_id = model_id  # no mapping
            # sample_model_id = self.img2shapes_mapping[model_id][
            #     random.randint(0, len(self.img2shapes_mapping[model_id]) - 1)] # coarse mapping
            if sample_model_id not in self.model_id_list:
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

        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'ShapeNetImageDataset'
