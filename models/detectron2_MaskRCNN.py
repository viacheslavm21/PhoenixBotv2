# python -m pip install detectron2 -f \ https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

import sys

import os

#sys.path.append('detectron2')
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog


def get_model(weights_path, device):
    # mock dataset registration:
    #register_coco_instances("CBR1_train", {},
    #                        'data/ChargerBot-room1-full-train.json',
    #                        'data')

    keypoint_names = ['UL', 'UR', 'BL', 'BR']
    keypoint_flip_map = [('UL', 'UR'), ('BL', 'BR')]
    keypoint_connection_rules = [('UL', 'UR', (100, 200, 50)), ('UR', 'BR', (100, 200, 50)),
                                 ('BR', 'BL', (100, 200, 50)), ('BL', 'UL', (100, 200, 50))]

    MetadataCatalog.get("CBR1_train").keypoint_names = keypoint_names
    MetadataCatalog.get("CBR1_train").keypoint_flip_map = keypoint_flip_map
    MetadataCatalog.get("CBR1_train").keypoint_connection_rules = keypoint_connection_rules
    # configure inference:
    # Inference should use the config with parameters that are used in training
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("CBR1_train",)
    if device != 'cpu':
        pass
    else:
        cfg.MODEL.DEVICE = device
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4  # p rev 2. Four is at the boundary of GPU memory limit
    cfg.SOLVER.BASE_LR = 0.004  # pick a good LR. prev : 0.00025
    cfg.SOLVER.MAX_ITER = 1500  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # faster, and good enough for this toy dataset (default: 512) prev: 128
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 4
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:

    # MASK-RCNN CHEKPOINT PATH
    cfg.MODEL.WEIGHTS = os.path.join(weights_path)  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    return predictor

