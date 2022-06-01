# all necessary classes and functions that implement ResNet50 model, train, testing, inference, visualizing
# vision-main is the snapshot of torchvision repo from github. We will need some references from there.
# if the repo will be uploaded to github, we should consider connecting it as subrepo, or delete everything not needed

# if ran as __main__, initiates training of the network.

import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
import sys
sys.path.append('models/vision-main/references/detection')

import os, json, cv2, copy, io, numpy as np, math, time

from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
import torchvision.models.detection.mask_rcnn

import albumentations as A # Library for augmentations

import transforms, utils, engine, train
import utils
from utils import collate_fn
from coco_utils import get_coco_api_from_dataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util

from contextlib import redirect_stdout

# augs

def train_transform():
    return A.Compose([
        A.Sequential([
            A.Rotate (limit=30, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.75),
            A.RandomBrightnessContrast(brightness_limit=0.01, contrast_limit=0.01, brightness_by_max=True, always_apply=False, p=1), # Random change of brightness & contrast
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'), # More about keypoint formats used in albumentations library read at https://albumentations.ai/docs/getting_started/keypoints_augmentation/
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels']) # Bboxes should have labels, read more here https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    )


class ClassDataset(Dataset):
    def __init__(self, root, annos, split, transform=None, demo=False):
        self.root = root
        with open(annos) as json_file:
            data = json.load(json_file)
            self.annos = np.asarray(data['annotations'])[split]
            self.imgs = \
            np.asarray([os.path.join(self.root, img_dict['file_name']) for img_dict in np.asarray(data['images'])])[
                split]
        self.transform = transform
        self.demo = demo

        """        # delete bad images
        bad = [
            'content/socket_v2/frame51_augmented_order_32.jpg',
            'content/socket_v2/frame60_augmented_order_8.jpg',
            'content/socket_v2/frame60_augmented_order_22.jpg',
            'content/socket_v2/frame62_augmented_order_37.jpg',
            'content/socket_v2/frame60_augmented_order_6.jpg',
            'content/socket_v2/frame52_augmented_order_15.jpg',
            'content/socket_v2/frame51_augmented_order_2.jpg',
            'content/socket_v2/frame61_augmented_order_5.jpg',
            'content/socket_v2/frame62_augmented_order_0.jpg',
        ]

        for bad_name in bad:
            indx = np.where(self.imgs == bad_name)
            np.delete(self.imgs,indx)
            np.delete(self.annos,indx)
        """
        # modify bboxes

        self.bboxes = []
        for indx in range(len(self.annos)):
            bboxes_original = [self.annos[indx]['bbox']]
            bboxes_original[0][2] += bboxes_original[0][0]
            bboxes_original[0][3] += bboxes_original[0][1]
            self.bboxes.append(bboxes_original)

    def __getitem__(self, idx):

        img_path = self.imgs[idx]
        # img_original = Image.open(img_path).convert("RGB")

        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

        # bboxes_original = [self.annos[idx]['bbox']]
        # bboxes_original[0][2]+=bboxes_original[0][0]
        # bboxes_original[0][3]+=bboxes_original[0][1]

        bboxes_original = self.bboxes[idx]
        # All objects are glue tubes
        bboxes_labels_original = ['Socket' for _ in bboxes_original]

        keypoints_original = self.annos[idx]['keypoints']
        keypoints_original = [np.asarray(keypoints_original).reshape(-1, 3)]
        # print('keypoints_original', keypoints_original)
        # mask = np.array([True,True,False,True,True,False,True,True,False,True,True,False,True,True,False,True,True,False,True,True,False,True,True,False,True,True,False,True,True,False,True,True,False,True,True,False,True,True,False])

        if self.transform:
            # Converting keypoints from [x,y,visibility]-format to [x, y]-format + Flattening nested list of keypoints
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2]
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]
            # print('keypoints_original_flattened', keypoints_original_flattened)

            # Apply augmentations
            # try to apply aug:
            try:
                transformed = self.transform(image=img_original, bboxes=bboxes_original,
                                             bboxes_labels=bboxes_labels_original, keypoints=keypoints_original_flattened)
                img = transformed['image']
                bboxes = transformed['bboxes']
                # print("transformed['keypoints']",transformed['keypoints'])
                # Unflattening list transformed['keypoints']
                # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
                # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2], where each keypoint is in [x, y]-format
                # Then we need to convert it to the following list:
                # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]]
                #print(f"'{img_path}',")
                keypoints_transformed_unflattened = np.reshape(np.array(transformed['keypoints']), (-1, 19, 2)).tolist()

                # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
                keypoints = []
                for o_idx, obj in enumerate(keypoints_transformed_unflattened):  # Iterating over objects
                    obj_keypoints = []
                    for k_idx, kp in enumerate(obj):  # Iterating over keypoints in each object
                        # kp - coordinates of keypoint
                        # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
                        obj_keypoints.append(kp + [keypoints_original[o_idx][k_idx][2]])
                    keypoints.append(obj_keypoints)
            except Exception as e:
                #print(f'Exception {e}. Apply no augmentation to the image')
                img, bboxes, keypoints = img_original, bboxes_original, keypoints_original

        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original

            # Convert everything into a torch tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64)  # all objects are glue tubes
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)
        img = F.to_tensor(img)

        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor([1 for _ in bboxes_original],
                                                    dtype=torch.int64)  # all objects are glue tubes
        target_original["image_id"] = torch.tensor([idx])
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (
                    bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor(keypoints_original, dtype=torch.float32)
        img_original = F.to_tensor(img_original)

        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target

    def __len__(self):
        return len(self.imgs)

# this is the main function that implements RESNET. Can use pretrained models.

def get_model(num_keypoints, weights_path=None, load_device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512),
                                       aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes=2,
                                                                   # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)

    if weights_path:
        state_dict = torch.load(weights_path, map_location=load_device)
        model.load_state_dict(state_dict)

    return model

### We MUST redefine class from cocotools API, to set a different oks_sigmas

class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        if not isinstance(iou_types, (list, tuple)):
            raise TypeError(f"This constructor expects iou_types of type list or tuple, instead  got {type(iou_types)}")
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]
            #print(iou_type)
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            #print(dir(coco_eval.cocoDt))
            if iou_type == 'keypoints':
              coco_eval.params.kpt_oks_sigmas = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]) # IMPORTANT FOR NUMBER OF KEYPOINTS
            img_ids, eval_imgs = evaluate_step(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate_step(imgs):
    with redirect_stdout(io.StringIO()):
        #print('imgs',imgs)
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    print('Here we go')
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    #print(dir(coco))
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    #print(dir(coco))
    #print(coco.getCatIds())

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        #print(res[0]['keypoints'][0])
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)


if __name__ == '__main__':
    # Train!

    DATASET_LEN = 1025
    TRAIN_SIZE = 950
    IMGS_FOLDER = '../data/datasets/socketCoco'
    ANNOS_FILE = '../data/datasets/socketCoco/extended.json'

    NUM_KEYPOINTS = 19 # BE AWARE (EDIT NOT ONLY HERE, but also oks sigmas in CocoEvaluator class, unflattened keypoints vector len!!!!)
    num_epochs = 55
    BATCH_SIZE = 4

    # create split
    split = np.full(DATASET_LEN, False)
    split[:TRAIN_SIZE] = True
    np.random.shuffle(split)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset_train = ClassDataset(root=IMGS_FOLDER, annos=ANNOS_FILE, split=split, transform=train_transform(), demo=False)
    dataset_test = ClassDataset(root=IMGS_FOLDER, annos=ANNOS_FILE, split=~split, transform=None, demo=False)

    data_loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_model(num_keypoints=NUM_KEYPOINTS)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0012, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    # print(evaluate(model, data_loader_test, device))

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1000)
        #print('Im done training')
        lr_scheduler.step()
        #print('Go val')
        evaluate(model, data_loader_test, device)
        #print('Im done val')
        if epoch%10 == 0:
            torch.save(model.state_dict(), f'outputs/keypointsrcnn_weights_3_epoch{epoch}.pth')

    # Save model weights after training
    torch.save(model.state_dict(), 'outputs/keypointsrcnn_weights_3_final.pth')



