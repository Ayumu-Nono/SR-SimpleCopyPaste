import numpy as np
from PIL import Image, ImageDraw
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

ia.seed(3)

image = Image.open('img/sample.jpg')
width: int = 384
height: int = 256
image = image.resize((width, height))
image_np = np.array(image)

# Define an example bbox
bbs = BoundingBoxesOnImage([
    BoundingBox(x1=300, y1=40, x2=350, y2=80),
    BoundingBox(x1=230, y1=160, x2=300, y2=190)
], shape=image_np.shape)

print(bbs.shape)
print(bbs)
# Define our augmentation pipeline
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_px={"x": (-50, 50), "y": (-50, 50)})
], )


# Augment BBs and images.
image_aug, bbs_aug = seq(image=image_np, bounding_boxes=bbs)

# print coordinates before/after augmentation (see below)
# use .x1_int, .y_int, ... to get integer coordinates
for i in range(len(bbs.bounding_boxes)):
    before = bbs.bounding_boxes[i]
    after = bbs_aug.bounding_boxes[i]
    print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
        i,
        before.x1, before.y1, before.x2, before.y2,
        after.x1, after.y1, after.x2, after.y2)
    )

# image with BBs before/after augmentation (shown below)
image_before = bbs.draw_on_image(image_np, size=2)
image_before = Image.fromarray(image_before)
image_before.save('img/bbs_before.jpg')

image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
image_after = Image.fromarray(image_after)
image_after.save('img/bbs_after.jpg')


