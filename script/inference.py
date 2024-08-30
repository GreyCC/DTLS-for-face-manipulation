import sys
import os
sys.path.append(".")
sys.path.append("..")

from dtls_smiling import DTLS, Trainer
import torch.cuda
from torchvision import transforms
import cv2

from PIL import Image, ImageOps
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms, utils

from util.unet_smiling import Unet
import numpy as np

dtls_size=512
device = "cuda:0" if torch.cuda.is_available() else "cpu"
load_path="./myDTLS_young_1e-2/model_last.pt"
test_input="fake_dataset_128/0/1.png"

#dir = '../restyle-encoder/edit_result_group/qqq/0.png'
#dir = '../restyle-encoder/real_image/_crops/zcy-1.jpg'
dir = "fake_dataset_128/0/20.png"
out_dir = "./tmp_folder/"

label = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5,5]
#out_dir = "./edit_result/zcy-1/"
#test_input="Test/put_your_photos_here/10.png"
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=3,
    residual=False
).to(device)
dtls = DTLS(
    model,
    image_size = dtls_size,
    device=device,
).to(device)

print("Loading : ", load_path)
data = torch.load(load_path, map_location=device)

dtls.load_state_dict(data['model'], strict=False)

def inference(img, t):
    '''img = cv2.imread(img, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = torch.clamp((img * 255.0).round(), 0, 255) / 255.
    img = img.unsqueeze(0).to(device)
    img = img * 2 - 1'''
    blur_img, img_sr = dtls.sample(batch_size=1, t=t,label = label, img=img.unsqueeze(0),device=device)
    img_sr = (img_sr + 1) / 2
    blur_img = (blur_img + 1) / 2
    utils.save_image(img_sr, str(f'inference_output.png'), nrow=1)
    return img_sr

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def get_smile_faces(image_address):
    if is_image_file(image_address):
        image = cv2.imread(image_address, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        image = cv2.resize(image, (128, 128))
        image = torch.from_numpy(np.transpose(image[:, :, [2, 1, 0]], (2, 0, 1))).float()
        image = torch.clamp((image * 255.0).round(), 0, 255) / 255.
        #image = image.unsqueeze(0).to(device)
        #image = image * 2 - 1
        img = image.unsqueeze(0).to(device)
        #img = img * 2 - 1
        with torch.no_grad():
            #image = image.unsqueeze(0).to(device)
            img = image.clone()
            image = image * 2 - 1
            temp = image.clone()
            out_lis = []
            for t in range(0,11):
                if t == 0:
                    out_lis.append(to_pil_image(img.squeeze(0), mode=None))
                    continue
                #print(temp.unsqueeze(0).shape)
                out = inference(temp.unsqueeze(0), t * 0.5)
                out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
                #print(out.squeeze(0).shape)
                out_lis.append(to_pil_image(out.squeeze(0), mode=None))

        torch.cuda.empty_cache()
        #out = out.clamp(0, 1).cpu().data
        # path = f'Test/output/{os.path.basename(image_address[:-4])}_P2S.png'
        # vutils.save_image(out, path, normalize=True, scale_each=True, nrow=1)
        return out_lis
    
res_lis = get_smile_faces(dir)
cnt = 0
os.makedirs(out_dir, exist_ok=True)
for img in res_lis:
    img.save(f"{out_dir}{cnt}.png")
    cnt = cnt + 1
#Image.fromarray(np.array(res_lis)).save(f'{dir}/result.png')

total_width = sum(img.width for img in res_lis)
max_height = max(img.height for img in res_lis)
new_image = Image.new('RGB', (total_width, max_height))
x_offset = 0
for img in res_lis:
    new_image.paste(img, (x_offset, 0))
    x_offset += img.width
new_image.save(f"{out_dir}/result.png")