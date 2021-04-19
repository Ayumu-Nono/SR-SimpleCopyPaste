from typing import List, Tuple
from PIL import Image, ImageDraw
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

im = Image.open('img/sample.jpg')
im_bk = Image.open("img/background.jpg")

width: int = 384
height: int = 256

im = im.resize((width, height))
im_bk = im_bk.resize((width, height))
im_bk.save("img/background2.jpg")

print(im.size)

mask = Image.new("L", im.size, 0)
draw = ImageDraw.Draw(mask)
# # polygons: List[Tuple[int, int]] = [(100, 100), (150, 40), (300, 200)]
polygons: List[Tuple[int, int]] = [(170, 220), (170, 20), (380, 10), (380, 230)]
draw.polygon(polygons, fill=255)
# im_maked = Image.composite(im, im_bk, mask)
# im_maked.save('img/masked.jpg')

# 変換器の設定
seq_aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_px={"x": (-100, 100), "y": (-100, 100)})
], random_state=1)

im_array = np.array(im)
aug_img = seq_aug.augment_image(im_array)
aug_img_jpg = Image.fromarray(aug_img)
aug_img_jpg.save('img/augimg.jpg')

print(mask)
mask_array = np.array(mask)
aug_mask = seq_aug.augment_image(mask_array)
aug_mask_jpg = Image.fromarray(aug_mask)
aug_mask_jpg.save('img/augmask.jpg')


# mask = Image.new("L", im.size, 0)
# draw = ImageDraw.Draw(mask)
# # polygons: List[Tuple[int, int]] = [(100, 100), (150, 40), (300, 200)]
# polygons: List[Tuple[int, int]] = [(170, 220), (170, 20), (380, 10), (380, 230)]
# draw.polygon(polygons, fill=255)
# im_maked = Image.composite(im, im_bk, mask)
# im_maked.save('img/masked.jpg')

