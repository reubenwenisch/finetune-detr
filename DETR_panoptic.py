from PIL import Image
import requests
import io
import math
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import numpy
torch.set_grad_enabled(False)

import panopticapi
from panopticapi.utils import id2rgb, rgb2id
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
mode = 'val'

# These are the COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Detectron2 uses a different numbering scheme, we build a conversion table
coco2d2 = {}
count = 0
for i, c in enumerate(CLASSES):
  if c != "N/A":
    coco2d2[i] = count
    count+=1

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
model = model.to(device)
postprocessor = postprocessor.to(device)
model.eval();

url = "http://images.cocodataset.org/val2017/000000281759.jpg"
im = Image.open(requests.get(url, stream=True).raw)

im
img = transform(im).unsqueeze(0)
out = model(img.to(device))

# compute the scores, excluding the "no-object" class (the last one)
scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
# threshold the confidence
keep = scores > 0.85

# Plot all the remaining masks
ncols = 5
fig, axs = plt.subplots(ncols=ncols, nrows=math.ceil(keep.sum().item() / ncols), figsize=(18, 10))
for line in axs:
    for a in line:
        a.axis('off')
for i, mask in enumerate(out["pred_masks"][keep]):
    ax = axs[i // ncols, i % ncols]
    ax.imshow(mask.cpu(), cmap="cividis")
    ax.axis('off')
fig.tight_layout()

# the post-processor expects as input the target size of the predictions (which we set here to the image size)
result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

import itertools
import seaborn as sns
palette = itertools.cycle(sns.color_palette())

# The segmentation is stored in a special-format png
panoptic_seg = Image.open(io.BytesIO(result['png_string']))
panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8).copy()
# We retrieve the ids corresponding to each mask
panoptic_seg_id = rgb2id(panoptic_seg)

# Finally we color each mask individually
panoptic_seg[:, :, :] = 0
for id in range(panoptic_seg_id.max() + 1):
  panoptic_seg[panoptic_seg_id == id] = numpy.asarray(next(palette)) * 255
plt.figure(figsize=(15,15))
plt.imshow(panoptic_seg)
plt.axis('off')
plt.show()

def save_image_panoptic(result, file_name):
  # The segmentation is stored in a special-format png
  panoptic_seg = Image.open(io.BytesIO(result['png_string']))
  panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8).copy()
  # We retrieve the ids corresponding to each mask
  panoptic_seg_id = rgb2id(panoptic_seg)
  # Finally we color each mask individually
  panoptic_seg[:, :, :] = 0
  # for id in range(panoptic_seg_id.max() + 1):
  #   panoptic_seg[panoptic_seg_id == id] = numpy.asarray(next(palette)) * 255
  im = Image.fromarray(panoptic_seg)
  im.save(f'./panoptic/panoptic_{mode}2017/' + file_name)

result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]
# result["segments_info"]

from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torchvision.transforms import ToTensor
# path2data = "/home/reuben/Atom360/Learning/data/dataset/images"
path2data = "/home/wenisch/Atom360/AI/Learning/data/dataset/images"
path2json_train = "./annotations/train.json"
path2json_test = "./annotations/test.json"
coco_train_dset = dset.CocoDetection(root = path2data, annFile = path2json_train, transform = ToTensor())
coco_test_dset = dset.CocoDetection(root = path2data, annFile = path2json_test, transform = ToTensor())

train_dataloader = DataLoader(coco_train_dset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(coco_test_dset, batch_size=1, shuffle=False)

from pycocotools.coco import COCO
from src.atom_seg import create_panoptic_annotation_format, create_image_panpotic_annotation

coco_train = COCO(path2json_train)
coco_test = COCO(path2json_test)

n_images = 0

if mode =='train':
    loader = train_dataloader
else:
    loader = test_dataloader

def images_annotations_info(loader):
    global n_images
    annotations = []
    images = []
    for image, labels in loader:
        # img = transform(im).unsqueeze(0)
        #todo
        out = model(image.to(device))
        result = postprocessor(out, torch.as_tensor(image.shape[-2:]).unsqueeze(0))[0]
        # image_id = [label['image_id'] for label in labels]
        image_id = int(labels[0]['image_id'])
        print(f'processing {image_id}')
        # result = result.cpu()
        if mode == 'train':
            file_name = coco_train.loadImgs(ids=int(image_id))[0]['file_name']
            width = coco_train.loadImgs(ids=int(image_id))[0]['width']
            height = coco_train.loadImgs(ids=int(image_id))[0]['height']
        else:
            file_name = coco_test.loadImgs(ids=int(image_id))[0]['file_name']
            width = coco_test.loadImgs(ids=int(image_id))[0]['width']
            height = coco_test.loadImgs(ids=int(image_id))[0]['height']
        save_image_panoptic(result, file_name)
        image = create_image_panpotic_annotation(file_name, width, height, image_id)
        images.append(image)
        annotations.append(create_panoptic_annotation_format(image_id, file_name, result))
        n_images += 1
    return images, annotations

from src.atom_seg import get_coco_json_panoptic_format, create_category_annotation
import json
# Label ids of the dataset
category_ids = {
    # "outlier": 0,
    "rebar": 1,
    "spall": 2,
    "crack": 3,
}

for keyword in ["train"]:
    coco_format = get_coco_json_panoptic_format()
    # Create category section
    coco_format["categories"] = create_category_annotation(category_ids)
    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"] = images_annotations_info(loader)
    with open(f"panoptic/annotations/panoptic_{mode}2017.json".format(keyword),"w") as outfile:
        json.dump(coco_format, outfile)
    print("Created %d annotations for images in folder: %s" % (n_images, 'panoptic'))