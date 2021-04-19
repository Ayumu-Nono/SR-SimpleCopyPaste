import numpy as np
from PIL import Image, ImageDraw
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

ia.seed(3)

image = Image.open('img/sample.jpg')
im_bk = Image.open("img/background.jpg")
width: int = 384
height: int = 256
image = image.resize((width, height))
im_bk = im_bk.resize((width, height))
image_np = np.array(image)
bg_np = np.array(im_bk)

# Define an example bbox
bbs = BoundingBoxesOnImage([
    BoundingBox(x1=300, y1=40, x2=350, y2=80),
    BoundingBox(x1=230, y1=160, x2=300, y2=190)
], shape=image_np.shape)

# Define our augmentation pipeline
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale=(0.8,1.25),
        translate_percent=(-0.3,0.3))
])


# Augment BBs and images.
image_aug, bbs_aug = seq(image=image_np, bounding_boxes=bbs)
bg_aug = seq(image=bg_np)

# 3) Dealing with bounding boxes outside of the image
bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

# print coordinates before/after augmentation (see below)
# use .x1_int, .y_int, ... to get integer coordinates
# for i in range(len(bbs.bounding_boxes)):
#     before = bbs.bounding_boxes[i]
#     after = bbs_aug.bounding_boxes[i]
#     print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
#         i,
#         before.x1, before.y1, before.x2, before.y2,
#         after.x1, after.y1, after.x2, after.y2)
#     )

# image with BBs before/after augmentation (shown below)
# image_before = bbs.draw_on_image(image_np, size=2)
# image_before = Image.fromarray(image_before)
# image_before.save('img/bbs_before.jpg')

bg_aug = Image.fromarray(bg_aug)
bg_aug.save('img/background_aug.jpg')


# The adjusted ground-truth annotations
image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
image_after = Image.fromarray(image_after)
image_after.save('img/bbs_after3.jpg')

# bbox -> mask
mask_after = Image.new("L", image_after.size, 0)
draw = ImageDraw.Draw(mask_after)
# We have not implemented random selection yet
# if bbs.bounding_boxes is not None:
for i in range(len(bbs.bounding_boxes)):
    after = bbs_aug.bounding_boxes[i]
    draw.rectangle([after.x1, after.y1, after.x2, after.y2], fill=255)
mask_after.save('img/mask_aug_bbox3.jpg')
image_aug = Image.fromarray(image_aug)

# CopyPaste
masked_after = Image.composite(image_aug, bg_aug, mask_after)
masked_after.save('img/copy_paste_bbox3.jpg')    