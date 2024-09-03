import torch
from util.unet_smiling import UNet_v2, discriminator_conditioned
from dtls_smiling import DTLS, Trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--mode', default='train', type=str, help="mode for either 'train' or 'eval'")
parser.add_argument('--size', default=128, type=int, help="size of HR image")
parser.add_argument('--train_steps', default=200001, type=int)
parser.add_argument('--lr_rate', default=2e-5, help="learning rate")
parser.add_argument('--sample_every_iterations', default=2000, type=int, help="sample SR images for every number of iterations")
parser.add_argument('--save_folder', default="Test_gan_2e-4", type=str, help="Folder to save your train or evaluation result")
parser.add_argument('--load_path', type=str, help="None or directory to pretrained model")
# parser.add_argument('--data_path', default='/hdda/Datasets/Face_super_resolution/images1024x1024/', type=str, help="directory to your training dataset")
parser.add_argument('--data_path', default='/hdda/Datasets/fake_dataset_young_128_0.5', type=str, help="directory to your training dataset")
parser.add_argument('--batch_size', default=2, type=int)
args = parser.parse_args()

device = args.device if torch.cuda.is_available() else "cpu"

model = UNet_v2(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=3,
).to(device)

dis = discriminator_conditioned(dim=64, dim_mults=(1, 2, 4, 8),channels=3).to(device)

print("unet and discriminator ok")

dtls = DTLS(
    model,
    image_size = args.size,
    device=device,
).to(device)

print("dtls ok")

trainer = Trainer(
    dtls,
    dis,
    args.data_path,
    image_size = args.size,
    train_batch_size = args.batch_size,
    train_lr = args.lr_rate,
    train_num_steps = args.train_steps, # total training steps
    gradient_accumulate_every = 1,      # gradient accumulation steps
    ema_decay = 0.995,                  # exponential moving average decay
    fp16 = False,                       # turn on mixed precision training with apex
    results_folder = args.save_folder,
    load_path = args.load_path,
    device = device,
    save_and_sample_every = args.sample_every_iterations
)
print("trainer ok")

if args.mode == 'train':
    trainer.train()
elif args.mode == 'eval':
    trainer.evaluation()
