import torch

DATA_DIR  = "data"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
TRAIN_CSV  = f"{DATA_DIR}/train.csv"
PLOTS_DIR = "plots"
BEST_MODEL_PATH = "best_model.pth"
BASELINE_MODEL_PATH = "baseline_resnet18.pth"

IMAGE_SIZE = 224
RESIZE_TO = 256
BATCH_SIZE = 32
NUM_WORKERS = 2
TEST_SIZE = 0.20
RANDOM_STATE = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

NUM_CLASSES = 5

CLASS_NAMES = [
    "Cassava Bacterial Blight",       # 0
    "Cassava Brown Streak Disease",   # 1
    "Cassava Green Mottle",           # 2
    "Cassava Mosaic Disease",         # 3
    "Healthy",                        # 4
]

LABEL_SMOOTHING = 0.1
CLASS_WEIGHT_POWER = 0.5

PHASE1_EPOCHS = 5
PHASE1_LR = 3e-4
PHASE1_WD = 1e-4

PROGRESSIVE_STAGES = [
    # (layer_name, description,   lr,    n_epochs)
    ("layer4", "Layer4 only",  5e-5, 7),
    ("layer3", "Layer4+3", 3e-5, 7),
    ("layer2", "Layer4+3+2", 1e-5, 7),
]

EARLY_STOPPING_PATIENCE = 5
MAX_GRAD_NORM = 1.0

CUTMIX_ALPHA = 0.3

BASELINE_EPOCHS = 8
BASELINE_LR = 1e-3

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
