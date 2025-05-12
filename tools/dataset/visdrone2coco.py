import os
import cv2
from tqdm import tqdm
import json
 
 
def convert_to_cocodetection(dir, output_dir):
    train_dir = os.path.join(dir, "VisDrone2019-DET-train")
    val_dir = os.path.join(dir, "VisDrone2019-DET-val")
    train_annotations = os.path.join(train_dir, "annotations")
    val_annotations = os.path.join(val_dir, "annotations")
    train_images = os.path.join(train_dir, "images")
    val_images = os.path.join(val_dir, "images")
    id_num = 0
 
    categories = [
            {"id": 0, "name": "ignored regions", "supercategory": "nark"},
            {"id": 1, "name": "pedestrian", "supercategory": "nark"},
            {"id": 2, "name": "people", "supercategory": "nark"},
            {"id": 3, "name": "bicycle", "supercategory": "nark"},
            {"id": 4, "name": "car", "supercategory": "nark"},
            {"id": 5, "name": "van", "supercategory": "nark"},
            {"id": 6, "name": "truck", "supercategory": "nark"},
            {"id": 7, "name": "tricycle", "supercategory": "nark"},
            {"id": 8, "name": "awning-tricycle", "supercategory": "nark"},
            {"id": 9, "name": "bus", "supercategory": "nark"},
            {"id": 10, "name": "motor", "supercategory": "nark"},
            {"id": 11, "name": "others", "supercategory": "nark"}
        ]
    for mode in ["train", "val"]:
        images = []
        annotations = []
        print(f"start loading {mode} data...")
        if mode == "train":
            set = os.listdir(train_annotations)
            annotations_path = train_annotations
            images_path = train_images
        else:
            set = os.listdir(val_annotations)
            annotations_path = val_annotations
            images_path = val_images
        for idx, i in enumerate(tqdm(set)):
            f = open(annotations_path + "/" + i, "r")
            name = i.replace(".txt", "")
            image = {}
            height, width = cv2.imread(images_path + "/" + name + ".jpg").shape[:2]
            file_name = name + ".jpg"
            image["file_name"] = file_name
            image["height"] = height
            image["width"] = width
            image["id"] = idx
            images.append(image)
            for line in f.readlines():
                annotation = {}
                line = line.replace("\n", "")
                if line.endswith(","):  # filter data
                    line = line.rstrip(",")
                line_list = [int(i) for i in line.split(",")]
                bbox_xywh = [line_list[0], line_list[1], line_list[2], line_list[3]]
                annotation["image_id"] = idx
                annotation["bbox"] = bbox_xywh
                annotation["category_id"] = int(line_list[5])
                annotation["id"] = id_num
                annotation["iscrowd"] = 0
                annotation["segmentation"] = []
                annotation["area"] = bbox_xywh[2] * bbox_xywh[3]
                id_num += 1
                annotations.append(annotation)
        dataset_dict = {}
        dataset_dict["images"] = images
        dataset_dict["annotations"] = annotations
        dataset_dict["categories"] = categories
        json_str = json.dumps(dataset_dict)
        with open(f'{output_dir}/VisDrone2019-DET_{mode}_coco.json', 'w') as json_file:
            json_file.write(json_str)
    print("json file write done...")
 
 
def get_test_namelist(dir, out_dir):
    full_path = out_dir + "/" + "test.txt"
    file = open(full_path, 'w')
    for name in tqdm(os.listdir(dir)):
        name = name.replace(".txt", "")
        file.write(name + "\n")
    file.close()
    return None
 
 
def centerxywh_to_xyxy(boxes):
    """
    args:
        boxes:list of center_x,center_y,width,height,
    return:
        boxes:list of x,y,x,y,cooresponding to top left and bottom right
    """
    x_top_left = boxes[0] - boxes[2] / 2
    y_top_left = boxes[1] - boxes[3] / 2
    x_bottom_right = boxes[0] + boxes[2] / 2
    y_bottom_right = boxes[1] + boxes[3] / 2
    return [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
 
 
def centerxywh_to_topleftxywh(boxes):
    """
    args:
        boxes:list of center_x,center_y,width,height,
    return:
        boxes:list of x,y,x,y,cooresponding to top left and bottom right
    """
    x_top_left = boxes[0] - boxes[2] / 2
    y_top_left = boxes[1] - boxes[3] / 2
    width = boxes[2]
    height = boxes[3]
    return [x_top_left, y_top_left, width, height]
 
 
def clamp(coord, width, height):
    if coord[0] < 0:
        coord[0] = 0
    if coord[1] < 0:
        coord[1] = 0
    if coord[2] > width:
        coord[2] = width
    if coord[3] > height:
        coord[3] = height
    return coord
 
 
if __name__ == '__main__':
    convert_to_cocodetection(r"/mnt/d/tinydetection/datasets/visdrone",r"/mnt/d/tinydetection/datasets/visdrone/annotations_coco")