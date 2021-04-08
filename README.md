# Simple Copy Paste

## ストラテジー

1. 画像を表示
2. アノテーションを読み込む
3. アノテーションを可視化
4. 切り取り
5. 貼り付け

- マスクごと線形変換（平行移動・左右反転）をできるようにする。
- 線形変換した後のマスクを背景画像に貼り付けられるようにする
- 線形変換した後にアノテーションのポリゴンの位置が変わるはずなので、それも同じように移動されるようにする

commit
1. We randomly select two images and apply random scale jittering and random horizontal flipping on each of them. 
    random scale jittering: 
    
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_px={"x": (-500, 500), "y": (-200, 200)})
    random horizontal flipping:
    iaa.Fliplr(0.5)
    
2. Then we select a random subset of objects from one of the images and paste them onto the other image.



3. Lastly, we adjust the ground-truth annotations accordingly: we remove fully occluded objects and update the masks and bounding boxes of partially occluded objects.