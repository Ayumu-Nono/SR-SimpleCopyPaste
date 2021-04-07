from typing import List, Tuple
from PIL import Image, ImageDraw

im = Image.open('img/sample.jpg')
im_bk = Image.open("img/background.jpg")

width: int = 384
height: int = 256

im = im.resize((width, height))
im_bk = im_bk.resize((width, height))

im_bk.save("img/background2.jpg")
mask = Image.new("L", im.size, 0)
draw = ImageDraw.Draw(mask)
polygons: List[Tuple[int, int]] = [(100, 100), (150, 40), (300, 200)]
draw.polygon(polygons, fill=255)
im_maked = Image.composite(im, im_bk, mask)
im_maked.save('img/masked.jpg')

