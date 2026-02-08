# config.py

# model
# MODEL_NAME = "resnet18"   # "resnet50" or "resnet101"
MODEL_NAME = "resnet34"   # "resnet50" or "resnet101"

NUM_CLASSES = 10
STUDENT_HOOK_LAYER = "layer3"

# training
BATCH_SIZE = 128
# EPOCHS = 10
EPOCHS = 100
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# system
NUM_WORKERS = 4
DEVICE = "cuda"

# checkpoint
# SAVE_DIR = "./checkpoints/train/resnet18/"
SAVE_DIR = "./checkpoints/train/resnet34/"
SAVE_LAST_NAME = "last.pth"
SAVE_BEST_NAME = "best.pth"
