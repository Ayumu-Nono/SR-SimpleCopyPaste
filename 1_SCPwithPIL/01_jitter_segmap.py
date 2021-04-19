import numpy as np
from PIL import Image, ImageDraw
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


ia.seed(3)

image = Image.open('img/sample.jpg')
width: int = 384
height: int = 256
image = image.resize((width, height))
image_np = np.array(image)
# print(image_np.shape)

# Define an example segmentation map (int32, 128x128).
# Here, we arbitrarily place some squares on the image.
# Class 0 is our intended background class.
segmap = np.zeros((height, width, 1), dtype=np.int32)
segmap[40:90, 320:360, 0] = 1
segmap[30:80, 240:290, 0] = 2
segmap[180:190, 230:290, 0] = 3
segmap = SegmentationMapsOnImage(segmap, shape=image_np.shape)

# Define our augmentation pipeline
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_px={"x": (-50, 50), "y": (-50, 50)})
],)

# Augment images and segmaps.
image_aug, segmap_aug = seq(image=image_np, segmentation_maps=segmap)

# image with segmaps before/after augmentation
image_before = segmap.draw_on_image(image_np)[0]
image_before = Image.fromarray(image_before)
image_before.save('img/segmap_before.jpg')

image_after = segmap_aug.draw_on_image(image_aug)[0]
image_after = Image.fromarray(image_after)
image_after.save('img/segmap_after.jpg')
