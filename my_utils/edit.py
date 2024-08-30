import math
import torch.cuda
from util.unet_smiling import Unet
# from unet_old import Unet
# from dtls import DTLS, Trainer
from dtls_smiling import DTLS, Trainer
import argparse
import time
import os
from torchvision import transforms, utils

parser = argparse.ArgumentParser()
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--size', default=128, type=int, help="size of HR image")
parser.add_argument('--save_folder', default="results/hr_inpaint/simple_momentum_128", type=str, help="Folder to save your train or evaluation result")
parser.add_argument('--load_path', type=str, help="None or directory to pretrained model")
parser.add_argument('--data_path', default='images1024x1024', type=str, help="directory to your training dataset")
args = parser.parse_args()

device = args.device if torch.cuda.is_available() else "cpu"
    

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=3,
    residual=False
).to(device)

dtls = DTLS(
    model,
    image_size = args.size,
    device=device,
).to(device)

def Inference():
    data,label = next(iter(self.dl))
    data.to(self.device)
    input_img_set = torch.tensor([])
    out_img_set = torch.tensor([])
    FFHQ_quality_MANIQA = 0
    FFHQ_quality_NIQE = 0
    for image in data:
        input_img, out_img = self.ema_model.sample(batch_size=1, img=image.unsqueeze(0),label=[float(row[0]) for row in label],device=self.device)
        out_img = (out_img + 1) / 2
        input_img = (input_img + 1) /2
        utils.save_image(out_img, str(self.results_folder / f'temp.png'), nrow=self.nrow)

        input_img_set = torch.cat((input_img_set, input_img.to("cpu")), dim=0)
        out_img_set = torch.cat((out_img_set, out_img.to("cpu")), dim=0)

        NIQE_mark = self.niqe(str(self.results_folder / f'temp.png')).item()
        MANIQA_mark = self.MANIQA(str(self.results_folder / f'temp.png')).item()

        FFHQ_quality_MANIQA += MANIQA_mark
        FFHQ_quality_NIQE += NIQE_mark

        os.remove(str(self.results_folder / f'temp.png'))

    FFHQ_quality_MANIQA /= self.batch_size
    FFHQ_quality_NIQE /= self.batch_size

    img_set = torch.cat((input_img_set, out_img_set), dim=0)
    utils.save_image(img_set, str(self.results_folder / f'{self.step}_FFHQ.png'), nrow=input_img_set.shape[0])

    
    self.metrics_list.append(f"FFHQ Images MANIQA: {FFHQ_quality_MANIQA} | NIQE: {FFHQ_quality_NIQE}")
    
    file = open(f"{self.results_folder}/quality.txt", 'w')
    for line in self.metrics_list:
        file.write(line + "\n")
    file.close()


'''trainer = Trainer(
    dtls,
    args.data_path,
    image_size = args.size,
    train_batch_size = args.batch_size,
    train_lr = args.lr_rate,
    train_num_steps = args.train_steps, # total training steps
    gradient_accumulate_every = 2,      # gradient accumulation steps
    ema_decay = 0.995,                  # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
    input_image = args.input_image,
    device = device,
    save_and_sample_every = args.sample_every_iterations
)

if args.mode == 'train':
    trainer.train()
elif args.mode == 'eval':
    trainer.evaluation()
'''