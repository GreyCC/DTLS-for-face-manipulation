import copy
import torch.nn.functional as F
import numpy as np
import glob
import shutil
import cv2
import os
import errno
import torch
import lpips
import pyiqa
import math
import random
import torchvision
import wandb

from torch import nn
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

####### helpers functions

def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class DTLS(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        device,
    ):
        super().__init__()
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.device = device
        self.MSE_loss = nn.MSELoss()
        self.smiling_factor = 1
        
    @torch.no_grad()
    def sample(self, img=None, t=None):
        R_x = self.denoise_fn(img, t)
        return R_x

    
    def p_losses(self, x_start, t, label, device):
        x_recon = []
        loss = 0
        for idx in label[:-1]:
            step = torch.full((x_start.shape[0],), idx, dtype=torch.float).to(self.device)
            tmp_res = self.denoise_fn(x_start[:,label.index(idx),:,:,:].to(device), step)
            loss = loss + self.MSE_loss(tmp_res, x_start[:,label.index(idx) + 1,:,:,:].to(device))
            x_recon.append(tmp_res.clone())
        return loss, x_recon

    def forward(self, x, label, *args, **kwargs):
        b, n, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.tensor(random.choices(range(0, len(label) - 1), k=b)).to(device)
        return self.p_losses(x, t, [float(row[0]) for row in label], device, *args, **kwargs)

# dataset classes

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_size):
        self.root_dir = Path(root_dir)
        self.image_paths = []
        self.smiles = []

        self.img_list = []
        self.label_list = []
        for smile_folder in sorted(self.root_dir.iterdir(), key=lambda x: float(x.name)):
            self.label_list.append(smile_folder.name)
        for smile_folder in sorted(self.root_dir.iterdir(), key=lambda x: float(x.name)):
            if smile_folder.is_dir():
                for image_path in smile_folder.glob('*png'):
                    self.img_list.append(image_path.name)
                break
        self.label_list = sorted(self.label_list, key=lambda x: float(x))
        
        self.transform = transforms.Compose([
            #transforms.Resize((int(image_size*1.1), int(image_size*1.1))),
            #transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        transformed_images = []
        for label in self.label_list:
            image_path = self.root_dir / label / img_name
            img = Image.open(image_path)
            transformed_img = self.transform(img)
            transformed_images.append(transformed_img)
        stacked_images = torch.stack(transformed_images, dim=0)
        return stacked_images, self.label_list


# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        discriminator,
        folder,
        *,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = None,
        load_path = None,
        shuffle=True,
        device,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)

        self.image_size = diffusion_model.image_size

        self.discriminator = discriminator

        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.nrow = 4
        self.metrics_list = []

        self.ds = Dataset(folder, image_size)

        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=shuffle, pin_memory=True, num_workers=train_batch_size))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=(0.9, 0.999), eps=1e-8)
        self.opt_d = Adam(self.discriminator.parameters(), lr=train_lr, betas=(0.9, 0.999), eps=1e-8)

        self.BCE_loss = torch.nn.BCELoss(size_average=True,reduction='none')

        self.step = 0

        self.device = device
        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.reset_parameters()


        self.best_quality = 0
        if load_path is not None:
            self.load(load_path)

        wandb.init(project="DTLS_face_manipulation")


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)


    def save_last(self):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'dis': self.discriminator.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model_last.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path, map_location=self.device)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema_model.load_state_dict(data['ema'], strict=False)
        self.discriminator.load_state_dict(data['dis'], strict=False)

    def train(self):
        backwards = partial(loss_backwards, self.fp16)
        self.step = 0
        while self.step < self.train_num_steps:
            data, label = next(iter(self.dl))
            data.to(self.device)

            overall_gan = 0
            overall_dis_real = 0
            overall_dis_fake = 0

            self.opt_d.zero_grad()
            for j in range(data.size(1)):
                image = data[:, j, :, :, :].to(self.device)
                t = torch.ones(self.batch_size).to(self.device) * float(label[j][0])
                score_true = self.discriminator(image, t)
                GAN_true = torch.ones_like(score_true)
                overall_dis_real = overall_dis_real + self.BCE_loss(score_true, GAN_true)
            overall_dis_real = overall_dis_real / data.size(1)
            backwards(overall_dis_real, self.opt_d)

            loss, x_recon = self.model(data.to(self.device), label)
            for j in range(len(x_recon)):
                t = torch.ones(self.batch_size).to(self.device) * float(label[j+1][0])
                img = x_recon[j]
                score_false = self.discriminator(img.detach(), t)
                GAN_false = torch.zeros_like(score_false)
                overall_dis_fake = overall_dis_fake + self.BCE_loss(score_false, GAN_false)
            overall_dis_fake = overall_dis_fake / len(x_recon)
            backwards(overall_dis_fake, self.opt_d)
            self.opt_d.step()

            self.opt.zero_grad()
            for j in range(len(x_recon)):
                t = torch.ones(self.batch_size).to(self.device) * float(label[j+1][0])
                img = x_recon[j]
                score_fake = self.discriminator(img, t)
                GAN_fake = torch.ones_like(score_fake)
                overall_gan = overall_gan + self.BCE_loss(score_fake, GAN_fake) * 2e-4
            overall_gan = overall_gan / len(x_recon)
            loss = loss / len(x_recon)
            backwards((loss + overall_gan), self.opt)
            self.opt.step()

            wandb.log({"MSE loss": loss.item(), "GAN loss": overall_gan.item(),
                       "Dis real": overall_dis_real.item(), "Dis false": overall_dis_fake.item()})

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step == 0 or self.step % self.save_and_sample_every == 0:
                data, label = next(iter(self.dl))
                input = data[:,0,:,:].to(self.device)
                save_gt = input.clone().to("cpu")
                ind = 0
                list_result = input.clone()
                for t in label[:-1]:
                    ind +=1
                    t = torch.ones(input.shape[0]).to(self.device) * float(t[0])
                    print("Referencing: ", t)
                    input = self.ema_model.sample(input, t)
                    list_result = torch.cat((list_result, input), dim=0)
                    save_gt = torch.cat((save_gt, data[:,ind,:,:]), dim=0)

                utils.save_image(save_gt.add(1).mul(0.5), str(self.results_folder / f'{self.step}_GT.png'), nrow=data.shape[0])
                utils.save_image(list_result.add(1).mul(0.5), str(self.results_folder / f'{self.step}_samples.png'), nrow=data.shape[0])
                wandb.log({"Ground truth": wandb.Image(str(self.results_folder / f'{self.step}_GT.png'))})
                wandb.log({"Checkpoint result": wandb.Image(str(self.results_folder / f'{self.step}_samples.png'))})
                self.save_last()

            self.step += 1
        print('training completed')
