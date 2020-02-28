import mask_rcnn_utils as utils
import mask_rcnn as modellib
import os
import sys
import json
import datetime
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa



VAL_IMAGE_PATH = '/home/jjonathanmak/cs271proj/Semantic_Segmentation_Dataset/validation/images'

VAL_IMAGE_IDS = sorted(os.listdir(VAL_IMAGE_PATH))

COCO_WEIGHTS_PATH = "/home/jjonathanmak/cs271proj/mask_rcnn_coco.h5"

class EyeSegmentationConfig(utils.Config):
    """Configuration for training on the OpenEDS segmentation dataset."""
    NAME = "eye_segmentation"

    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 4  # pupil, iris, sclera, background

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    DETECTION_MIN_CONFIDENCE = 0.5

    # Backbone network architecture
    # experiment with resnet50 and resnet 101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 128
    # IMAGE_MAX_DIM = 512
    # IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB) (?)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400

class EyeSegmentationInferenceConfig(EyeSegmentationConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

# Use configuation from NucleusConfig, but override
# image resizing so we see the real sizes here
class EyeSegmentationNoResizeConfig(EyeSegmentationConfig):
    IMAGE_RESIZE_MODE = "none"

def train(model, dataset_train, dataset_val):
    """Train the model."""

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])
    
    augmentation = []

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=model.config.LEARNING_RATE,
                epochs=5,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=model.config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='all')

def train_mrcnn(dataset_train, dataset_val, load_last=False):
    """Train Mask R-CNN Model on nuclei segmentation dataset."""

    # Create model
    config = EyeSegmentationConfig()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir='logs')

    # Select weights file to load
    if load_last:
        weights_path = model.find_last()
    else:
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)

    # Load weights
    if load_last:
        model.load_weights(weights_path, by_name=True)
    else:
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

    # Train
    train(model, dataset_train, dataset_val)

    # Inference
    config = NucleusInferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='logs')
    weights_path = model.find_last()
    model.load_weights(weights_path, by_name=True)

    return model
