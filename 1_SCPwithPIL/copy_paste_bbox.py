import numpy as np
from PIL import Image, ImageDraw
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

ia.seed(3)

def copy_paste_bb(im_path, im_bg_path, bbs=None, width=None, height=None):
    # load images
    image = Image.open(im_path)
    im_bg = Image.open(im_bg_path)
    if width is not None and height is not None:
        width: int = width
        height: int = height
    else:
        width: int = image.width
        height: int = image.height
    image = image.resize((width, height))
    im_bg = im_bg.resize((width, height))
    image_np = np.array(image)
    bg_np = np.array(im_bg)
    
    # load the bbox
    if bbs is not None:
        bbs = bbs
    else:
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

    # Dealing with bounding boxes outside of the image
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

    # Draw the mask from bbox
    mask_after = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_after)
    # We have not implemented random selection yet
    # if bbs.bounding_boxes is not None:
    for i in range(len(bbs.bounding_boxes)):
        after = bbs_aug.bounding_boxes[i]
        draw.rectangle([after.x1, after.y1, after.x2, after.y2], fill=255)
    
    # CopyPaste
    image_aug = Image.fromarray(image_aug)
    bg_aug = Image.fromarray(bg_aug)
    im_cp = Image.composite(image_aug, bg_aug, mask_after)
    
    return im_cp

# Implementation
im_path = 'img/sample.jpg'
im_bg_path = 'img/background.jpg'
im_cp = copy_paste_bb(im_path=im_path, im_bg_path=im_bg_path, width=384, height=256)
im_cp.save('img/copy_paste_bbs.jpg') 
