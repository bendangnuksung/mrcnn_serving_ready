import numpy as np


class mask_config():
    def __init__(self, NUMBER_OF_CLASSES):
        self.NAME = "tags"
        self.IMAGES_PER_GPU = 2
        self.NUM_CLASSES = 1 + NUMBER_OF_CLASSES  # Background + tags
        self.STEPS_PER_EPOCH = 100
        self.DETECTION_MIN_CONFIDENCE = 0.9
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 1
        self.NAME = None  # Override in sub-classes
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 1
        self.STEPS_PER_EPOCH = 1000
        self.VALIDATION_STEPS = 50
        self.BACKBONE = "resnet101"
        self.COMPUTE_BACKBONE_SHAPE = None
        self.BACKBONE_STRIDES = [4, 8, 16, 32, 64]
        self.FPN_CLASSIF_FC_LAYERS_SIZE = 1024
        self.TOP_DOWN_PYRAMID_SIZE = 256
        self.RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
        self.RPN_ANCHOR_RATIOS = [0.5, 1, 2]
        self.RPN_ANCHOR_STRIDE = 1
        self.RPN_NMS_THRESHOLD = 0.7
        self.RPN_TRAIN_ANCHORS_PER_IMAGE = 256
        self.POST_NMS_ROIS_TRAINING = 2000
        self.POST_NMS_ROIS_INFERENCE = 1000
        self.USE_MINI_MASK = True
        self.MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
        self.IMAGE_RESIZE_MODE = "square"
        self.IMAGE_MIN_DIM = 800
        self.IMAGE_MAX_DIM = 1024
        self.IMAGE_MIN_SCALE = 0
        self.MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
        self.TRAIN_ROIS_PER_IMAGE = 200
        self.ROI_POSITIVE_RATIO = 0.33
        self.POOL_SIZE = 7
        self.MASK_POOL_SIZE = 14
        self.MASK_SHAPE = [28, 28]
        self.MAX_GT_INSTANCES = 100
        self.RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
        self.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
        self.DETECTION_MAX_INSTANCES = 100
        self.DETECTION_MIN_CONFIDENCE = 0.7
        self.DETECTION_NMS_THRESHOLD = 0.3
        self.LEARNING_RATE = 0.001
        self.LEARNING_MOMENTUM = 0.9
        self.WEIGHT_DECAY = 0.0001
        self.LOSS_WEIGHTS = {"rpn_class_loss": 1., "rpn_bbox_loss": 1., "mrcnn_class_loss": 1., "mrcnn_bbox_loss": 1.,
                             "mrcnn_mask_loss": 1.}
        self.USE_RPN_ROIS = True
        self.TRAIN_BN = False  # Defaulting to False since batch size is often small
        self.GRADIENT_CLIP_NORM = 5.0

        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, 3])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES
