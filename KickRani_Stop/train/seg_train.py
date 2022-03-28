import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory
ROOT_DIR = os.path.abspath("../..")
sys.path.append(ROOT_DIR)

# Module 불러오기
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from mrcnn.config import Config
from mrcnn import utils
from mrcnn.model import MaskRCNN

# Directory to save logs and model checkpoints
# Log 저장할 경로 지정
DEFAULT_LOGS_DIR = os.path.join('/content/drive/MyDrive/coco_person/', "logs")

############################################################
#  Configurations
############################################################

class PersonConfig(Config):
    """
    Configuration for training on Person data.
    Derives from the base Config class and overrides values specific
    to the Person dataset.
    """
    # Recognizable name for configuration
    NAME = "person"

    # GPU 당 이미지 개수
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs
    GPU_COUNT = 1

    # 클래스 개수 (배경 포함)
    NUM_CLASSES = 2  # Background + category(person)

    USE_MINI_MASK = True

    # 이미지, json 파일 경로 지정
    train_img_dir = "/content/Mask_RCNN/dataset/train/image"
    train_json_path = "/content/Mask_RCNN/dataset/train/train.json"
    valid_img_dir = "/content/Mask_RCNN/dataset/train/image"
    valid_json_path = "/content/Mask_RCNN/dataset/val/val.json"

############################################################
#  Dataset
############################################################
class PersonDataset(utils.Dataset):
    def load_coco(self, image_dir, json_path, class_ids=None,
                  class_map=None, return_coco=False):
        """
        Load the Person dataset.
        """

        coco = COCO(json_path)

        # 클래스 load 하기
        if not class_ids:
            # 모든 클래스에 대해서 class id 구하기
            class_ids = sorted(coco.getCatIds())

        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # 중복 제거
            image_ids = list(set(image_ids))
        else:
            # 모든 이미지에 대해서 image id 구하기
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("person", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "person", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_keypoint(self, image_id):
        """
        Load keypoints from the given image
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "person":
            return super(PersonDataset, self).load_mask(image_id)

        instance_keypoints = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        for annotation in annotations:
            class_id = self.map_source_class_id(
                "person.{}".format(annotation['category_id']))
            if class_id:
                keypoint = annotation['keypoints']

                instance_keypoints.append(keypoint)
                class_ids.append(class_id)

        keypoints = np.stack(instance_keypoints, axis=1)
        class_ids = np.array(class_ids, dtype=np.int32)
        return keypoints, class_ids

    def load_mask(self, image_id):
        """
        Load instance masks for the given image.
        Converts the different mask format to a form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with one mask per instance
        class_ids: a 1D array of class IDs of the instance masks
        """
        # COCO 형식의 이미지가 아니면, 부모 클래스에 위임.
        image_info = self.image_info[image_id]
        if image_info["source"] != "person":
            return super(PersonDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        # [height, width, instance_count] 모양의 mask 생성
        # Mask의 각 channel에 대응하는 ClassId 리스트 생성
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "person.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])

                # 1 픽셀 미만의 아주 작은 object는 skip하기
                if m.max() < 1:
                    continue
                # Iscrowd에 해당하면, negative class ID 사용.
                if annotation['iscrowd']:
                    class_id *= -1
                    # annToMask()가 주어진 것보다 더 작은 차원의 mask를 리턴하면, mask resize
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # instance mask array로 변환
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # super class 호출하여 빈 mask 리턴
            return super(PersonDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """
        Return a link to the image in the COCO Website.
        """
        super(PersonDataset, self).image_reference(image_id)

    # Functions from pycocotools
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # merge all parts into one mask RLE code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # RLE
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

def train(model, config):
    """
    Train model
    """
    dataset_train = PersonDataset()
    dataset_train.load_coco(config.train_img_dir, config.train_json_path)
    dataset_train.prepare()

    dataset_valid = PersonDataset()
    dataset_valid.load_coco(config.valid_img_dir, config.valid_json_path)
    dataset_valid.prepare()

    model.train(dataset_train, dataset_valid,
                learning_rate=config.LEARNING_RATE,
                epochs=300,
                layers='3+') #3+ layers: 네트워크 전체 학습

############################################################
#  main
############################################################

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath("./")
    DEFAULT_LOGS_DIR = os.path.join('/content/drive/MyDrive/coco_person/', "logs")
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN for Person data.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="train")
    parser.add_argument('--weights', required=True,
                        metavar="path to weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    """
    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    """

    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    if args.command == "train":
        # Configuration
        config = PersonConfig()
        # Create model
        model = MaskRCNN(mode="training", config=config, model_dir=args.logs)
        # 불러올 weight 지정
        weights_path = args.weights
        # Weight 불러오기
        model.load_weights(weights_path, by_name=True)
        # Train
        train(model, config)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))

    config.display()
