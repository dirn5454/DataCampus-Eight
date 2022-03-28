import numpy as np

class Config(object):
    """Base configuration class.
    """
    # Configuration 이름
    NAME = None  # Override in sub-classes

    # GPU 개수 (CPU면 1)
    GPU_COUNT = 1

    # GPU당 학습할 이미지 개수
    IMAGES_PER_GPU = 2

    # training steps
    STEPS_PER_EPOCH = 1000

    # validation steps
    VALIDATION_STEPS = 50

    # Backbone network architecture
    BACKBONE = "resnet101"
    COMPUTE_BACKBONE_SHAPE = None

    # FPN Pyramid의 layer의 stride
    # Resnet101 backbone 기반
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # FC layers의 size
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Top-down layers의 size
    TOP_DOWN_PYRAMID_SIZE = 256

    # 클래스 개수 (배경 포함)
    NUM_CLASSES = 2

    # Square anchor side 길이
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Anchors의 비율(width/height)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    RPN_ANCHOR_STRIDE = 1
    RPN_NMS_THRESHOLD = 0.7
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    PRE_NMS_LIMIT = 6000

    # non-maximum suppression 이후의 ROI
    POST_NMS_ROIS_TRAINING = 2000  # Training
    POST_NMS_ROIS_INFERENCE = 1000  # Inference

    # Instance mask를 Resize
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width)

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 0
    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # 이미지 당 ROI의 개수
    TRAIN_ROIS_PER_IMAGE = 200

    # Positive ROIs의 비율
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Output mask의 shape
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances
    MAX_GT_INSTANCES = 100

    # RPN과 최종 detection의 Bounding box refinement standard deviation
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate
    LEARNING_RATE = 0.001

    # Learning Momentum
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    USE_RPN_ROIS = True
    TRAIN_BN = False
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                self.IMAGE_CHANNEL_COUNT])

        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")