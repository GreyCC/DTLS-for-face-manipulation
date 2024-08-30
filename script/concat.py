import cv2
from PIL import Image

import os

name = 'hh'

folder_path = f"./edit_result/{name}"
folder_path = f"../restyle-encoder/edit_result_group/{name}"

images = []

for i in range(5,11,1):
    file_path = os.path.join(folder_path, f"{i}.png")
    if os.path.exists(file_path):
        image = Image.open(file_path)
        images.append(image)
    else:
        print(f"File {i}.png not found.")


total_width = sum(img.width for img in images)
max_height = max(img.height for img in images)
new_image = Image.new('RGB', (total_width, max_height))
x_offset = 0
for img in images:
    new_image.paste(img, (x_offset, 0))
    x_offset += img.width
new_image.save(f"tmp/{name}1.png")