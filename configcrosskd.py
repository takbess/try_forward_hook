# configkd.py

# model
TEACHER_MODEL = "resnet34"
STUDENT_MODEL = "resnet18"
NUM_CLASSES = 10

# CrossKD
STUDENT_HOOK_LAYER = "layer3"
TEACHER_INJECT_LAYER = "layer4"
LAMBDA_CROSSKD = 1.0

# training
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# system
NUM_WORKERS = 4
DEVICE = "cuda"

# save
SAVE_DIR = "./checkpoints_kd"
SAVE_LAST = "student_crosskd_last.pth"
SAVE_BEST = "student_crosskd_best.pth"
teacher_load_from = "./checkpoints/train/resnet34/last.pth"
