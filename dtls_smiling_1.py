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

from torch import nn
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image
from util.discriminator import discriminator_v3 as d_3

from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr

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
        stochastic=False,
    ):
        super().__init__()
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.device = device
        self.MSE_loss = nn.MSELoss()
        
    @torch.no_grad()
    def sample(self, batch_size=16, img=None, t=None, imgname=None, label=None, device=None):
        if t == None:
            t = 5.9
        #print(label)
        blur_img = img[:,:,:,:].squeeze(0)
        #print(blur_img.shape)
        img_t = blur_img.clone()
        previous_x_s0 = None
        momentum = 0
        ####### Domain Transfer
        step = torch.full((batch_size,), t, dtype=torch.float).to(self.device)
        R_x = self.denoise_fn(img_t.to(device), step.to(device))
        img_t=R_x
        #print(img_t.shape)
        return blur_img, img_t
        
        while (t != label[it]):
            dir = -1 if t < label[it] else 1
            current_step = label[it]
            next_step = label[it + dir]
            print(f"Current Step of img: from {current_step} to {next_step}")

            #step = torch.full((batch_size,), label[it], dtype=torch.long).to(self.device)
            step = torch.full((batch_size,), label[it], dtype=torch.float).to(self.device)
            momentum_l = 0

            #if previous_x_s0 is None:
            #    momentum_l = 0
            #else:
            #    momentum_l = self.transform_func_sample(momentum, current_step)

            #weight = (1 - (current_step**2/self.image_size**2))
            #weight = (1 - math.log(current_step + 1 - self.size_list[-1])/math.log(self.image_size))

            #if previous_x_s0 is None:
            #    R_x = self.denoise_fn(img_t, step)
            #    # return blur_img, R_x
            #    previous_x_s0 = R_x
            #else:
            #    R_x = self.denoise_fn(img_t + momentum_l, step)
            R_x = self.denoise_fn(img_t.to(device), step.to(device))

            #momentum += previous_x_s0 - R_x
            previous_x_s0 = R_x

            # R_x = self.denoise_fn(img_t, step)

            # utils.save_image((R_x+1)/2, f"20230103_eval/{current_step}_SR.png")
            #x4 = self.transform_func_sample(R_x, next_step)
            img_t = R_x
            it = it + dir
        return blur_img, img_t
    
    def tensor2im(self, var):
        var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
        var = ((var + 1) / 2)
        var[var < 0] = 0
        var[var > 1] = 1
        var = var * 255
        return Image.fromarray(var.astype('uint8'))
    
    def p_losses(self, x_start, t, label, device):
        #print(f"x_start shape: {x_start.shape}")
        x_in = torch.empty(x_start.shape[:1] + x_start.shape[2:])
        x_out = torch.empty(x_start.shape[:1] + x_start.shape[2:])
        #print(f"x_out shape: {x_out.shape}")
        #print(f"lanel:{label}")
        t1 = []
        for i in range(t.shape[0]):
            #current_step = t[i]
            x_in[i] = x_start[i][t[i]].to(device)
            x_out[i] = x_start[i][t[i]+1].to(device)
            t1.append(label[t[i]])
        #print(torch.tensor(t1).float().to(device).shape)
        #print(x_out.to(device).shape)
        #print(x_start[:,label.index(0),:,:,:].shape)
        #print(x_in.shape)
        x_recon = self.denoise_fn(x_in.to(device), torch.tensor(t1).float().to(device))
        '''for i in label:
            lis = []
            for j in range(t.shape[0]):
                lis.append(i)
            tmp = self.denoise_fn(x_in.to(device), torch.tensor(lis).float().to(device))[0]
            img = self.tensor2im(tmp)
            img.save(f"./{i}.png")
        exit(0)'''
        loss = self.MSE_loss(x_recon, x_out.to(device))
        '''print(x_out[0].shape)
        img=self.tensor2im(x_out[0])
        img.save("./x_out.jpg")
        img=self.tensor2im(x_recon[0])
        img.save("./recon.jpg")
        print(x_start[:,label.index(0),:,:,:][0].shape)
        img=self.tensor2im(x_start[:,label.index(0),:,:,:][0])
        img.save("./x_start.jpg")
        ### Loss function
        print(loss)
        print(f"t:{t[0]}")
        print(f"t1:{t1[0]}")
        exit(0)'''
        return loss, x_recon

    def forward(self, x, label, *args, **kwargs):
        #print(label)
        b, n, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.tensor(random.choices(range(0, len(label) - 1), k=b)).to(device)
        #print(t)
        #t = torch.randint(1, self.num_timesteps + 1, (b,), device=device).long()
        return self.p_losses(x, t, [float(row[0]) for row in label], device, *args, **kwargs)


# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
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
        input_image,
        load_path = None,
        shuffle=True,
        device,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)

        self.image_size = diffusion_model.image_size

        self.discriminator = d_3(image_size=self.image_size, dim=24,
                                 dim_mults=(8, 4, 4, 2, 2, 1, 1),channels=3).to(device)
        # self.lpips = lpips.LPIPS(net='vgg').to(device)

        self.ema_model = copy.deepcopy(self.model)

        self.step_start_ema = step_start_ema

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.nrow = 4
        self.metrics_list = []
        self.input_image = input_image


        self.step = 0

        self.device = device
        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.reset_parameters()

        self.niqe = pyiqa.create_metric('niqe', device=torch.device(self.device))
        self.MANIQA = pyiqa.create_metric('maniqa', device=torch.device(self.device))


        if load_path != None:
            self.load(load_path)


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path, map_location=self.device)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema_model.load_state_dict(data['ema'], strict=False)
        self.discriminator.load_state_dict(data['dis'], strict=False)
    
    def evaluation(self):
        total_quality_MANIQA = 0
        total_quality_NIQE = 0
        total_img = 0
        blur_img_set = torch.tensor([])
        hq_img_set = torch.tensor([])

        # data = next(self.dl).to(self.device)
        # utils.save_image((data+1)/2, f"{self.results_folder}/True_hr.png")
        # data = F.interpolate(data, 32, mode="bilinear", antialias=True)
        # utils.save_image((data+1)/2, f"{self.results_folder}/True_lr.png")


        for idx, path in enumerate(sorted(glob.glob(os.path.join(self.input_image, '*')))):
            imgname = os.path.splitext(os.path.basename(path))[0]
            print(idx, imgname)
            # read image
            print(path)
            img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img = torch.clamp((img * 255.0).round(), 0, 255) / 255.
            img = img.unsqueeze(0).to(self.device)
            img = img * 2 - 1
            blur_img, img_sr = self.ema_model.sample(batch_size=1, img=img.unsqueeze(0),device=self.device)
            img_sr = (img_sr + 1) / 2
            blur_img = (blur_img + 1) / 2
            utils.save_image(img_sr, str(self.results_folder /  f'{imgname}.png'), nrow=1)
            utils.save_image(img_sr, str(self.results_folder /  f'temp.png'), nrow=self.nrow)

            NIQE_mark = self.niqe(str(self.results_folder / f'temp.png')).item()
            MANIQA_mark = self.MANIQA(str(self.results_folder / f'temp.png')).item()

            os.remove(str(self.results_folder / f'temp.png'))

            blur_img_set = torch.cat((blur_img_set, blur_img.to("cpu")), dim=0)
            hq_img_set = torch.cat((hq_img_set, img_sr.to("cpu")), dim=0)

            total_quality_MANIQA += MANIQA_mark
            total_quality_NIQE += NIQE_mark
            total_img +=1



        img_set = torch.cat((blur_img_set, hq_img_set), dim=0)
        utils.save_image(img_set, str(self.results_folder / f'{self.step}_overall.png'), nrow=blur_img_set.shape[0])
        utils.save_image(blur_img_set, str(self.results_folder / f'lq_overall.png'), nrow=6)
        utils.save_image(hq_img_set, str(self.results_folder / f'hq_overall.png'), nrow=6)

        print(f"Avg MANIQA: {total_quality_MANIQA / total_img}, NIQE: {total_quality_NIQE / total_img}")
        return total_quality_MANIQA / total_img, total_quality_NIQE / total_img
    
    def inference(self, img, t):
        blur_img_set = torch.tensor([])
        hq_img_set = torch.tensor([])

        # data = next(self.dl).to(self.device)
        # utils.save_image((data+1)/2, f"{self.results_folder}/True_hr.png")
        # data = F.interpolate(data, 32, mode="bilinear", antialias=True)
        # utils.save_image((data+1)/2, f"{self.results_folder}/True_lr.png")

        '''img = cv2.imread(image, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = torch.clamp((img * 255.0).round(), 0, 255) / 255.
        img = img.unsqueeze(0).to(self.device)
        img = img * 2 - 1'''
        blur_img, img_sr = self.ema_model.sample(batch_size=1, t=t, img=img.unsqueeze(0),device=self.device)
        img_sr = (img_sr + 1) / 2
        blur_img = (blur_img + 1) / 2
        utils.save_image(img_sr, str(f'tmp1.png'), nrow=1)
        return img_sr
        utils.save_image(img_sr, str(f'temp.png'), nrow=self.nrow)
