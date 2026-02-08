# config.py

# model
MODEL_NAME = "resnet34"   # "resnet50" or "resnet101"

NUM_CLASSES = 10

# training
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# system
NUM_WORKERS = 4
DEVICE = "cuda"

# checkpoint
SAVE_DIR = "./checkpoints"
SAVE_LAST_NAME = "student_last.pth"
SAVE_BEST_NAME = "student_best.pth"
