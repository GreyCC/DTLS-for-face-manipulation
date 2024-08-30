import sys
import os
sys.path.append(".")
sys.path.append("..")

from dtls_smiling_1 import DTLS, Trainer
import torch.cuda
from torchvision import transforms
import cv2

from PIL import Image, ImageOps
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import to_pil_image

from util.unet_smiling import Unet
import numpy as np

dtls_size=512
device = "cuda:0" if torch.cuda.is_available() else "cpu"
load_path="../test1/models/dtls.pt"
test_input="Test/smiling/1.png"

#dir = '../restyle-encoder/edit_result_group/qqq/0.png'
dir = '../restyle-encoder/real_image/_crops/zcy-1.jpg'
out_dir = "./edit_result/zcy-1/"
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

trainer = Trainer(
    dtls,
    image_size = dtls_size,
    load_path = load_path,
    input_image = test_input,
    device = device,
)

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
                out = trainer.inference(temp.unsqueeze(0), t*0.6)
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