import torch
import torch.nn as nn
import torchvision.models as models

resnet50 = models.resnet50(weights=None)
resnet101 = models.resnet101(weights=None)

resnet50.train()
resnet101.train()

saved_feat = {}

def r50_forward_hook(module, input, output):
    # output は grad を持った Tensor
    saved_feat["feat"] = output

# 例：layer2 の出力を取得
handle_r50 = resnet50.layer2.register_forward_hook(
    r50_forward_hook
)

def r101_pre_hook(module, input):
    # input は tuple
    # layer2 の入力を ResNet50 の layer2 出力で置換
    return (saved_feat["feat"],)

handle_r101 = resnet101.layer3.register_forward_pre_hook(
    r101_pre_hook
)

x = torch.randn(2, 3, 224, 224, requires_grad=True)
target = torch.randint(0, 1000, (2,))

criterion = nn.CrossEntropyLoss()

# forward
y50 = resnet50(x)     # hook で saved_feat["feat"] が埋まる
y101 = resnet101(x)   # pre-hook で feat が注入される

loss = criterion(y101, target)
loss.backward()

print(resnet50.layer2[0].conv1.weight.grad is not None)
print(resnet50.layer1[0].conv1.weight.grad is not None)
