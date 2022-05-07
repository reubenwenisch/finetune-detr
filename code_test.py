# " "/home/reuben/Atom360/Learning/data/dataset""
import os
cmd = 'python main.py --coco_path  "/home/wenisch/Atom360/AI/Learning/data/dataset"  \
        --coco_panoptic_path "panoptic" \
        --dataset_file "coco_panoptic" \
        --resume "weights/detr-r50-dc5-panoptic-da08f1b1.pth" \
        --lr 1e-5 \
        --batch_size 2 \
        --output_dir "output/panoptic/box_model" \
        --epochs 10'
print(os.getcwd())
os.system(cmd)