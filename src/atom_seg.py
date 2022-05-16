import json
import numpy as np
from pycocotools import mask
from skimage import measure
import os
import cv2
from PIL import Image
from panopticapi.utils import IdGenerator

annotation_id = 0
black = [0,0,0]

category_list = [
        {
            "supercategory": "rebar",
            "isthing": 1,
            "id": 1,
            "name": "rebar"
        },
        {
            "supercategory": "spall",
            "isthing": 1,
            "id": 2,
            "name": "spall"
        },
        {
            "supercategory": "crack",
            "isthing": 1,
            "id": 3,
            "name": "crack"
        }, 
        {
            "supercategory": "textile",
            "isthing": 0,
            "id": 92,
            "name": "banner"
        },
        {
            "supercategory": "textile",
            "isthing": 0,
            "id": 93,
            "name": "blanket"
        },
        {
            "supercategory": "building",
            "isthing": 0,
            "id": 95,
            "name": "bridge"
        },
        {
            "supercategory": "raw-material",
            "isthing": 0,
            "id": 100,
            "name": "cardboard"
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 107,
            "name": "counter"
        },
        {
            "supercategory": "textile",
            "isthing": 0,
            "id": 109,
            "name": "curtain"
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 112,
            "name": "door-stuff"
        },
        {
            "supercategory": "floor",
            "isthing": 0,
            "id": 118,
            "name": "floor-wood"
        },
        {
            "supercategory": "plant",
            "isthing": 0,
            "id": 119,
            "name": "flower"
        },
        {
            "supercategory": "food-stuff",
            "isthing": 0,
            "id": 122,
            "name": "fruit"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 125,
            "name": "gravel"
        },
        {
            "supercategory": "building",
            "isthing": 0,
            "id": 128,
            "name": "house"
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 130,
            "name": "light"
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 133,
            "name": "mirror-stuff"
        },
        {
            "supercategory": "structural",
            "isthing": 0,
            "id": 138,
            "name": "net"
        },
        {
            "supercategory": "textile",
            "isthing": 0,
            "id": 141,
            "name": "pillow"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 144,
            "name": "platform"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 145,
            "name": "playingfield"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 147,
            "name": "railroad"
        },
        {
            "supercategory": "water",
            "isthing": 0,
            "id": 148,
            "name": "river"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 149,
            "name": "road"
        },
        {
            "supercategory": "building",
            "isthing": 0,
            "id": 151,
            "name": "roof"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 154,
            "name": "sand"
        },
        {
            "supercategory": "water",
            "isthing": 0,
            "id": 155,
            "name": "sea"
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 156,
            "name": "shelf"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 159,
            "name": "snow"
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 161,
            "name": "stairs"
        },
        {
            "supercategory": "building",
            "isthing": 0,
            "id": 166,
            "name": "tent"
        },
        {
            "supercategory": "textile",
            "isthing": 0,
            "id": 168,
            "name": "towel"
        },
        {
            "supercategory": "wall",
            "isthing": 0,
            "id": 171,
            "name": "wall-brick"
        },
        {
            "supercategory": "wall",
            "isthing": 0,
            "id": 175,
            "name": "wall-stone"
        },
        {
            "supercategory": "wall",
            "isthing": 0,
            "id": 176,
            "name": "wall-tile"
        },
        {
            "supercategory": "wall",
            "isthing": 0,
            "id": 177,
            "name": "wall-wood"
        },
        {
            "supercategory": "water",
            "isthing": 0,
            "id": 178,
            "name": "water-other"
        },
        {
            "supercategory": "window",
            "isthing": 0,
            "id": 180,
            "name": "window-blind"
        },
        {
            "supercategory": "window",
            "isthing": 0,
            "id": 181,
            "name": "window-other"
        },
        {
            "supercategory": "plant",
            "isthing": 0,
            "id": 184,
            "name": "tree-merged"
        },
        {
            "supercategory": "structural",
            "isthing": 0,
            "id": 185,
            "name": "fence-merged"
        },
        {
            "supercategory": "ceiling",
            "isthing": 0,
            "id": 186,
            "name": "ceiling-merged"
        },
        {
            "supercategory": "sky",
            "isthing": 0,
            "id": 187,
            "name": "sky-other-merged"
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 188,
            "name": "cabinet-merged"
        },
        {
            "supercategory": "furniture-stuff",
            "isthing": 0,
            "id": 189,
            "name": "table-merged"
        },
        {
            "supercategory": "floor",
            "isthing": 0,
            "id": 190,
            "name": "floor-other-merged"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 191,
            "name": "pavement-merged"
        },
        {
            "supercategory": "solid",
            "isthing": 0,
            "id": 192,
            "name": "mountain-merged"
        },
        {
            "supercategory": "plant",
            "isthing": 0,
            "id": 193,
            "name": "grass-merged"
        },
        {
            "supercategory": "ground",
            "isthing": 0,
            "id": 194,
            "name": "dirt-merged"
        },
        {
            "supercategory": "raw-material",
            "isthing": 0,
            "id": 195,
            "name": "paper-merged"
        },
        {
            "supercategory": "food-stuff",
            "isthing": 0,
            "id": 196,
            "name": "food-other-merged"
        },
        {
            "supercategory": "building",
            "isthing": 0,
            "id": 197,
            "name": "building-other-merged"
        },
        {
            "supercategory": "solid",
            "isthing": 0,
            "id": 198,
            "name": "rock-merged"
        },
        {
            "supercategory": "wall",
            "isthing": 0,
            "id": 199,
            "name": "wall-other-merged"
        },
        {
            "supercategory": "textile",
            "isthing": 0,
            "id": 200,
            "name": "rug-merged"
        }]

panoptic_coco_categories = './panoptic_coco_categories.json'
with open(panoptic_coco_categories, 'r') as f:
    categories_list = json.load(f)
categories = {category['id']: category for category in categories_list}

panoptic_coco_categories = './panoptic_coco_categories_pan.json'
with open(panoptic_coco_categories, 'r') as f:
    categories_list_pan = json.load(f)
categories_pan = {category['id']: category for category in categories_list_pan}

id_generator = IdGenerator(categories)

def create_annotation_format(masks, category_id, image_id):
    global annotation_id
    annotation = {
            "segmentation": [],
            "area": [],
            "iscrowd": int(0),
            "image_id": int(image_id),
            "bbox": [],
            "category_id": int(category_id),
            "id": int(annotation_id)
        }
    ground_truth_binary_mask= cv2.copyMakeBorder(masks,1,1,1,1,cv2.BORDER_CONSTANT,value=black)
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)
    annotation["area"] = int(ground_truth_area)
    annotation["category_id"] = int(category_id)
    annotation["bbox"] = ground_truth_bounding_box.tolist()
    for contour in contours:
        contour = np.flip(contour, axis=1).astype(int)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)
    annotation_id += 1
    return annotation

def create_annotation_format2(masks, category_id, image_id):
    global annotation_id
    kernel = np.ones((2, 2), np.uint8)
    masks = cv2.dilate(masks, kernel, iterations=1)
    C, h = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    seg = [[float(x) for x in contour.flatten()] for contour in C]
    seg = [cont for cont in seg if len(cont) > 4]  # filter all polygons that are boxes
    rle = mask.frPyObjects(seg, masks.shape[0], masks.shape[1])
    annotation_id += 1
    return {
        'area': float(sum(mask.area(rle))),
        'bbox': list(mask.toBbox(rle).tolist()), #'bbox': list(mask.toBbox(rle)[0]),
        'category_id': int(category_id),
        'id': int(annotation_id),
        "image_id": int(image_id),
        'iscrowd': int(0),
        'segmentation': seg
    }

from .pycococreatortools import resize_binary_mask, binary_mask_to_polygon, binary_mask_to_rle

def create_annotation_info(image_id, category_id, binary_mask, 
                           image_size=None, tolerance=2, bounding_box=None):
    global annotation_id
    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask, image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)
    is_crowd = 0
    segmentation = binary_mask_to_polygon(binary_mask, tolerance)
    if not segmentation:
        return None
    annotation_id += 1
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": is_crowd,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    } 

    return annotation_info 

def mask_to_mscoco(alpha, annotations, img_id, mode='rle'):
        if mode == 'rle':
            in_ = np.reshape(np.asfortranarray(alpha), (alpha.shape[0], alpha.shape[1], 1))
            in_ = np.asfortranarray(in_)
            rle = mask.encode(in_)
            segmentation = rle[0]
        else:
            raise ValueError('Unknown mask mode "{}"'.format(mode))
        for idx, c in enumerate(np.unique(alpha)):
            area = mask.area(rle).tolist()
            if isinstance(area, list):
                area = area[0]
            bbox = mask.toBbox(rle).tolist()
            if isinstance(bbox[0], list):
                bbox = bbox[0]
            annotation = {
                'area': area,
                'bbox': bbox,
                'category_id': c,
                'id': len(annotations)+idx,
                'image_id': img_id,
                'iscrowd': 0,
                'segmentation': segmentation}
            annotations.append(annotation)
        return annotations 

def create_category_annotation(category_dict):
    global category_list
    # return categories_pan
    return category_list

def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name + '.jpg',
        "height": int(height),
        "width": int(width),
        "id": int(image_id)
    }
    return images

def create_image_panpotic_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name,
        "height": int(height),
        "width": int(width),
        "id": int(image_id)
    }
    return images

def get_coco_json_format():
    # Standard COCO format 
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format

def get_coco_json_panoptic_format():
    # Standard COCO format 
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format

def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

seg_id = 0

def get_color(cat_id):
    def random_color(base, max_dist=30):
        new_color = base + np.random.randint(low=-max_dist,
                                             high=max_dist+1,
                                             size=3)
        return tuple(np.maximum(0, np.minimum(255, new_color)))

def create_seg_info(result):
    segments_info_list = []
    # result_local = result.copy()
    for i, segment_info in enumerate(result): #["segments_info"]):
        semantic_id = segment_info['category_id']
        if semantic_id not in categories:
            print("semnantic id missing", semantic_id)
            continue
        segment_id, color = id_generator.get_id_and_color(segment_info['category_id'])
        segment_info["id"] = segment_id # seg_id #
        segment_info["iscrowd"] = int(0)
        labels = segment_info["category_id"]
        segments_info_list.append(segment_info)
    return segments_info_list

annotation_id_panoptic = 0

def create_panoptic_annotation_format(image_id, file_name, result, coco_detection):
    """Create Panoptic format dictionary

    Args:
        image_id (int): the id of the image for getting mode details about it
        file_name (string): name of the image being saved with panoptic details
        result (prediction dict): Output from the model

    Returns:
        _type_: _description_
    """
    segments_info = create_seg_info(result)
    annotation = {
        "file_name": file_name,
        "image_id": image_id,
    }
    # if segments_info != []:
    annotation["segments_info"] = segments_info
    segments_info_det, img_detection = convert_detection_to_panoptic_coco_format_single_core(coco_detection, image_id)
    for seg in segments_info_det:
        annotation["segments_info"].append(seg)
    return annotation, img_detection

# coco_detection = COCO(input_json_file)


def convert_detection_to_panoptic_coco_format_single_core(coco_detection, img_id):
    id_generator = IdGenerator(categories_pan)
    img = coco_detection.loadImgs(int(img_id))[0]
    pan_format = np.zeros((img['height'], img['width'], 3), dtype=np.uint8)
    overlaps_map = np.zeros((img['height'], img['width']), dtype=np.uint32)
    anns_ids = coco_detection.getAnnIds(img_id)
    anns = coco_detection.loadAnns(anns_ids)
    # panoptic_record = {}
    # panoptic_record['image_id'] = img_id
    file_name = '{}.png'.format(img['file_name'].rsplit('.')[0])
    # panoptic_record['file_name'] = file_name
    segments_info_det = []
    for ann in anns:
        if ann['category_id'] not in categories_pan:
            raise Exception('Panoptic coco categories file does not contain \
                category with id: {}'.format(ann['category_id'])
            )
        segment_id, color = id_generator.get_id_and_color(ann['category_id'])
        mask_det = coco_detection.annToMask(ann)
        overlaps_map += mask_det
        pan_format[mask_det == 1] = color
        ann.pop('segmentation')
        ann.pop('image_id')
        ann['id'] = segment_id
        segments_info_det.append(ann)
    if np.sum(overlaps_map > 1) != 0:
        raise Exception("Segments for image {} overlap each other.".format(img_id))
    # panoptic_record['segments_info'] = segments_info
    # annotations_panoptic.append(panoptic_record)
    img_detection = Image.fromarray(pan_format)
    return segments_info_det, img_detection