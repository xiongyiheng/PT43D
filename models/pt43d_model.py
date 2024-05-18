import os
from collections import OrderedDict

import numpy as np
import einops
import mcubes
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.transformer_networks.rand_transformer import RandTransformer
from models.networks.pvqvae_networks.auto_encoder import PVQVAE
from models.networks.clip_networks.network import CLIPImageEncoder

import utils.util
from utils.util_3d import init_mesh_renderer, render_sdf


class RandTransformerModel(BaseModel):
    def name(self):
        return 'Rand-Transformer-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()

        # -------------------------------
        # Define Networks
        # -------------------------------

        assert opt.tf_cfg is not None
        assert opt.vq_cfg is not None

        # load configs for tf and vq
        tf_conf = OmegaConf.load(opt.tf_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)

        # init tf model
        self.tf = RandTransformer(tf_conf, vq_conf=vq_conf)

        # init vqvae for decoding shapes
        mparam = vq_conf.model.params
        n_embed = mparam.n_embed
        embed_dim = mparam.embed_dim
        ddconfig = mparam.ddconfig

        n_down = len(ddconfig.ch_mult) - 1

        self.vqvae = PVQVAE(ddconfig, n_embed, embed_dim)
        self.load_vqvae(opt.vq_ckpt)
        self.vqvae.to(opt.device)
        self.vqvae.eval()

        # modify the tf's embedding to be the codebook learned from vqvae
        self.tf.embedding_encoder = nn.Embedding(n_embed, embed_dim)
        self.tf.embedding_encoder.load_state_dict(self.vqvae.quantize.embedding.state_dict())
        self.tf.embedding_encoder.requires_grad = False
        self.tf.to(opt.device)

        # init cond model
        self.cond_model = CLIPImageEncoder(model='ViT-B/32', device=opt.device)  # hard-coded
        for param in self.cond_model.parameters():
            param.requires_grad = True  # True  # fine-tune clip

        if self.isTrain:
            # ----------------------------------
            # define loss functions
            # ----------------------------------
            # self.criterion_nll = nn.NLLLoss()
            self.criterion_ce = nn.CrossEntropyLoss()
            self.criterion_ce.to(opt.device)

            # ---------------------------------
            # initialize optimizers
            # ---------------------------------
            self.optimizer = optim.Adam([p for p in self.tf.parameters() if p.requires_grad == True], lr=opt.lr)
            self.optimizer_cond = optim.Adam([p for p in self.cond_model.parameters() if p.requires_grad == True], lr=1e-5)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 10 if opt.dataset_mode == 'imagenet' else 10,
                                                       0.9)
            self.scheduler_cond = optim.lr_scheduler.StepLR(self.optimizer_cond, 10, 0.9)

            self.optimizers = [self.optimizer, self.optimizer_cond]
            self.schedulers = [self.scheduler, self.scheduler_cond]

            self.print_networks(verbose=False)

        # transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        resolution = tf_conf.data.resolution
        self.resolution = resolution

        # start token
        self.sos = 0
        self.counter = 0

        # init grid for lookup
        pe_conf = tf_conf.pe
        self.grid_size = pe_conf.zq_dim
        self.grid_table = self.init_grid(pos_dim=pe_conf.pos_dim, zq_dim=self.grid_size)

        # setup hyper-params 
        nC = resolution
        self.cube_size = 2 ** n_down  # patch_size
        self.stride = self.cube_size
        self.ncubes_per_dim = nC // self.cube_size
        assert nC == 64, 'right now, only trained with sdf resolution = 64'
        assert (nC % self.cube_size) == 0, 'nC should be divisable by cube_size'

        # setup renderer
        dist, elev, azim = 1.7, 20, 20
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.opt.device)

    def load_vqvae(self, vq_ckpt):
        assert type(vq_ckpt) == str
        state_dict = torch.load(vq_ckpt)

        self.vqvae.load_state_dict(state_dict)
        print(colored('[*] VQVAE: weight successfully load from: %s' % vq_ckpt, 'blue'))

    def init_grid(self, pos_dim=3, zq_dim=8):
        x = torch.linspace(-1, 1, zq_dim)
        y = torch.linspace(-1, 1, zq_dim)
        if pos_dim == 3:
            z = torch.linspace(-1, 1, zq_dim)
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
            grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
            pos_sos = torch.tensor([-1., -1., -1 - 2 / zq_dim]).float().unsqueeze(0)
        else:
            grid_x, grid_y = torch.meshgrid(x, y)
            grid = torch.stack([grid_x, grid_y], dim=-1)
            pos_sos = torch.tensor([-1., -1 - 2 / zq_dim]).float().unsqueeze(0)

        grid_table = grid.view(-1, pos_dim)
        grid_table = torch.cat([pos_sos, grid_table], dim=0)
        return grid_table

    def get_gen_order(self, sz, device):
        return torch.randperm(sz, device=device)

    def get_dummy_input(self, bs=1):

        ret = {}
        ret['sdf'] = torch.zeros(bs, 1, 64, 64, 64)  # .to(device)
        ret['idx'] = torch.zeros(bs, self.grid_size, self.grid_size,
                                 self.grid_size).long()  # .to(device) # grid size is8
        ret['z_q'] = torch.zeros(bs, 256, self.grid_size, self.grid_size, self.grid_size)  # .to(device)

        return ret

    def set_input(self, input=None, gen_order=None):

        self.x = input['sdf']
        self.x_idx = input['idx']
        # self.z_q = input['z_q']
        self.img = input['img']
        bs, dz, hz, wz = self.x_idx.shape
        # self.z_shape = self.z_q.shape

        # if self.opt.dataset_mode in ['pix3d_img', 'snet_img']:
        #     self.gt_vox = input['gt_vox']

        self.x_idx_seq = rearrange(self.x_idx, 'bs dz hz wz -> (dz hz wz) bs').contiguous()  # to (T, B)
        self.x_idx = self.x_idx_seq.clone()

        # prepare input for transformer
        T, B = self.x_idx.shape[:2]

        if gen_order is None:
            self.gen_order = self.get_gen_order(T, self.opt.device)
            self.context_len = -1  # will be specified in inference
        else:
            if len(gen_order) != T:

                self.context_len = len(gen_order)
                # pad the remaining
                remain = torch.tensor([i for i in range(T) if i not in gen_order]).to(gen_order)
                remain = remain[torch.randperm(len(remain))]
                self.gen_order = torch.cat([gen_order, remain])
            else:
                self.gen_order = gen_order

        x_idx_seq_shuf = self.x_idx_seq[self.gen_order]
        x_seq_shuffled = torch.cat([torch.LongTensor(1, bs).fill_(self.sos), x_idx_seq_shuf], dim=0)  # T+1
        pos_shuffled = torch.cat([self.grid_table[:1], self.grid_table[1:][self.gen_order]],
                                 dim=0)  # T+1, <sos> should always at start.

        self.inp = x_seq_shuffled[:-1].clone()
        self.tgt = x_seq_shuffled[1:].clone()
        self.inp_pos = pos_shuffled[:-1].clone()
        self.tgt_pos = pos_shuffled[1:].clone()

        self.counter += 1

        vars_list = ['gen_order',
                     'inp', 'inp_pos', 'tgt', 'tgt_pos',
                     'x_idx', 'x_idx_seq', 'x', 'img']

        self.tocuda(var_names=vars_list)

    def forward(self):
        """ given 
                inp, inp_pos, tgt_pos
            infer
                tgt
            outp is the prob. dist. over x_(t+1) at pos_(t+1)
            p(x_{t+1} | x_t, pos_t, pos_{t+1})
        """
        c_img = self.cond_model(self.img).float()
        self.outp = self.tf(self.inp, self.inp_pos, self.tgt_pos, c_img)

    def inference(self, data, seq_len=None, gen_order=None, topk=None, prob=None, alpha=0.2, should_render=False,
                  verbose=False):
        def top_k_logits(logits, k=30):
            v, ix = torch.topk(logits, k)
            out = logits.clone()
            out[out < v[:, :, [-1]]] = -float('Inf')
            return out

        self.tf.eval()
        self.cond_model.eval()
        # context: 
        #     - if prob is given, seq_len=1
        #     - else seq_len is defined by gen_order
        if prob is not None:
            if seq_len is None:
                seq_len = 1  # context
        else:
            if gen_order is None:
                if seq_len is None:
                    seq_len = 1  # context
            else:
                # if goes here, context_len will be given by gen_order
                # +1 to include sos
                seq_len = len(gen_order) + 1
        self.set_input(data, gen_order=gen_order)

        T = self.x_idx_seq.shape[0] + 1  # +1 since <sos>
        B = self.x_idx_seq.shape[1]

        if data['img'] is not None:
            c_img = self.cond_model(data['img']).float() # for eval

        if prob is not None:
            prob = prob[self.gen_order]
            prob = torch.cat([prob[:1], prob])
        with torch.no_grad():
            # auto-regressively gen
            pred = self.inp[:seq_len]
            for t in tqdm(range(seq_len, T), total=T - seq_len, desc='[*] autoregressively inferencing...'):
                inp = pred
                inp_pos = self.inp_pos[:t]
                tgt_pos = self.tgt_pos[:t]
                if c_img is not None:
                    outp = self.tf(inp, inp_pos, tgt_pos, c_img)
                else:
                    outp = self.tf(inp, inp_pos, tgt_pos)
                outp_t = outp[-1:]
                outp_t = F.log_softmax(outp_t, dim=-1)
                if prob is not None:
                    outp_t = (1 - alpha) * outp_t + alpha * prob[t:t + 1]
                    # outp_t = prob[t:t + 1]

                if topk is not None:
                    outp_t = top_k_logits(outp_t, k=topk)

                outp_t = F.softmax(outp_t, dim=-1)  # compute prob
                outp_t = rearrange(outp_t, 't b nc -> (t b) nc')
                pred_t = torch.multinomial(outp_t, num_samples=1).squeeze(1)
                pred_t = rearrange(pred_t, '(t b) -> t b', t=1, b=B)
                pred = torch.cat([pred, pred_t], dim=0)

            self.x = self.x
            pred = pred[1:][torch.argsort(self.gen_order)]  # exclude pred[0] since it's <sos>
            self.x_recon_tf = self.vqvae.decode_enc_idices(pred, z_spatial_dim=self.grid_size) # [:, 0:8]

        self.tf.train()
        self.cond_model.train()

    def single_view_recon(self, img_tensor, bs=1, topk=30):
        # get dummy data
        img_prob = None
        data = self.get_dummy_input(bs=bs)
        from utils.qual_util import get_img_prob
        data['img'] = img_tensor.to(self.opt.device)
        img_prob = get_img_prob(resnet2vq, data)
        for i in range(bs - 1):
            img_prob = torch.cat([img_prob, get_img_prob(resnet2vq, data)], dim=1)
        #.repeat(bs, 1, 1, 1).to(self.opt.device)
        data['img'] = img_tensor.repeat(bs, 1, 1, 1).to(self.opt.device)

        # self.inference(data, topk=30, prob=None, alpha=alpha)
        self.inference(data, prob=img_prob, seq_len=None, topk=topk, alpha=alpha)

        return self.x_recon_tf

    def eval_metrics(self, dataloader, thres=0.0):
        self.eval()

        ret = OrderedDict([
            ('iou', 0.0),
            ('iou_std', 0.0),
        ])
        self.train()
        return ret

    def backward(self):
        '''backward pass for the generator in training the unsupervised model'''
        target = rearrange(self.tgt, 'seq b -> (seq b)')  # 512, 10 -> 5120
        outp = rearrange(self.outp, 'seq b cls-> (seq b) cls')  # exclude the last one as its for <end> 512, 10, 512 -> 5120, 512

        loss_nll = self.criterion_ce(outp, target)

        self.loss = loss_nll

        self.loss_nll = loss_nll
        self.loss.backward()

    def optimize_parameters(self, total_steps):
        # self.vqvae.train()

        self.set_requires_grad([self.tf, self.cond_model], requires_grad=True)

        self.forward()
        self.optimizer.zero_grad()
        self.optimizer_cond.zero_grad()
        self.backward()
        self.optimizer.step()
        self.optimizer_cond.step()

    def get_logs_data(self):
        """ return a dictionary with
            key: graph name
            value: an OrderedDict with the data to plot
        
        """
        raise NotImplementedError
        return ret

    def get_current_errors(self):

        ret = OrderedDict([
            ('nll', self.loss_nll.data),
            # ('rec', self.loss_rec.data),
        ])

        return ret

    def get_current_visuals(self):

        with torch.no_grad():
            self.input = self.img
            self.image = render_sdf(self.renderer, self.x)
            # self.image_recon = render_sdf(self.renderer, self.x_recon)
            self.image_recon_tf = render_sdf(self.renderer, self.x_recon_tf)

        vis_tensor_names = [
            'input',
            'image',
            'image_recon_tf',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        # vis_tensor_names = ['%s/%s' % (phase, n) for n in vis_tensor_names]
        visuals = zip(vis_tensor_names, vis_ims)

        return OrderedDict(visuals)

    def save(self, label):

        state_dict = {
            'vqvae': self.vqvae.cpu().state_dict(),
            'tf': self.tf.cpu().state_dict(),
            'cond_model': self.cond_model.cpu().state_dict()
        }
        save_filename = 'rand_tf_%s.pth' % (label)
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(state_dict, save_path)
        self.vqvae.to(self.opt.device)
        self.tf.to(self.opt.device)
        self.cond_model.to(self.opt.device)

    def load_ckpt(self, ckpt):
        if type(ckpt) == str:
            state_dict = torch.load(ckpt)
        else:
            state_dict = ckpt
        self.vqvae.load_state_dict(state_dict['vqvae'])
        self.tf.load_state_dict(state_dict['tf'])
        self.cond_model.load_state_dict(state_dict['cond_model'])
        self.cond_model.required_grad = True
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))
