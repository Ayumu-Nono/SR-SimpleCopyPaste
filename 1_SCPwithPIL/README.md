# Simple Copy Paste with PIL

## ストラテジー

1. 画像を表示
2. アノテーションを読み込む
3. アノテーションを可視化
4. 切り取り
5. 貼り付け

- マスクごと線形変換（平行移動・左右反転）をできるようにする。
- 線形変換した後のマスクを背景画像に貼り付けられるようにする
- 線形変換した後にアノテーションのポリゴンの位置が変わるはずなので、それも同じように移動されるようにする

Method  
1. We randomly select two images and apply random scale jittering and random horizontal flipping on each of them.   
    random scale jittering:  
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_px={"x": (-500, 500), "y": (-200, 200)})  
    random horizontal flipping:  
        iaa.Fliplr(0.5)  
    
    bbox: https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html#a-simple-example  
    segmaps: https://imgaug.readthedocs.io/en/latest/source/examples_segmentation_maps.html#a-simple-example  

01_jitter_bbox.py : bbs_before.jpg -> bbs_after.jpg  
01_jitter_segmap.py : segmap_before.jpg -> segmap_after.jpg  

2. Then we select a random subset of objects from one of the images and paste them onto the other image.  

02_copy_paste_bbox.py : mask_aug_bbox.jpg & copy_paste_bbox.jpg

3. Lastly, we adjust the ground-truth annotations accordingly: we remove fully occluded objects and update the masks and bounding boxes of partially occluded objects.  

03_adjust_bbox.py  

## Next step
- bbox以外の対応(polygonなど、chicken checkのデータセットに合わせる)
- COCOformatに合わせる
- いくつかあるobjectsからrandomに選んで貼る

- 関数を作る
input 画像2枚 -> output 画像1枚