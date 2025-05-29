import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import yaml
from MlpNetwork import KernelPredictor2

#Use def "parse_model" to get model from yaml file.

# ----------------------------
# Utility Functions
# ----------------------------
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor

# ----------------------------
# CNN Modules
# ----------------------------
class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
    def forward(self, x):
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2],
                                    x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))

# ----------------------------
# Model Parser
# You can use this function for get model from yaml file.
# d : yaml file data
# ch : channel size
# example : "model = parse_model(d, ch=[3])"
# ----------------------------
def parse_model(d, ch):
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"]):  # backbone only (from yolov5)
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            args[j] = eval(a) if isinstance(a, str) else a

        n = max(round(n * d["depth_multiple"]), 1) if n > 1 else n
        if m in {Conv, Focus, C3}:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * d["width_multiple"], 8)
            args = [c1, c2, *args[1:]]
            if m is C3:
                args.insert(2, n)
                n = 1
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers)








# ----------------------------
# Load model.yaml
'''
with open("/home/okjy89/project/Adaptive_Kernel_Creater/Rooftop_Project/cnn.yaml", 'r') as f:
    d = yaml.safe_load(f)
model = parse_model(d, ch=[3])  # RGB 입력
'''
# ----------------------------













# ----------------------------
# ex


# ----------------------------
# Load Image
# ----------------------------
def load_image(path, img_size=640):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(img).unsqueeze(0)  # [1, 3, 640, 640]

# ----------------------------
# Run Backbone
# ----------------------------
if __name__ == "__main__":
    # Load model.yaml
    with open("/home/okjy89/project/Adaptive_Kernel_Creater/Rooftop_Project/cnn.yaml", 'r') as f:
        d = yaml.safe_load(f)
    model = parse_model(d, ch=[3])  # RGB 입력
    print("Model structure:", model)
    # Load image
    x = load_image("/home/okjy89/project/Adaptive_Kernel_Creater/Rooftop_Project/image4.jpeg")  # <-- 이미지 경로

    # Forward
    with torch.no_grad():
        y = model(x)
    print("Feature map shape:", y.shape)
    print("Feature map:", y)

    kernel_predictor = KernelPredictor2(Cin=512)
    with torch.no_grad():
        kernel = kernel_predictor(y)
        print("Dynamic kernel shape:", kernel.shape)
        print("Dynamic kernel:", kernel)


kernel_rgb = kernel.repeat(3, 1, 1, 1)  # [3, 1, 3, 3]
filtered = F.conv2d(x, kernel_rgb, padding=1, groups=3)


import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# 1. 원본 이미지 (x): [1, 3, 640, 640]
original = x.squeeze(0).detach().cpu()         # [3, H, W]
filtered = filtered.squeeze(0).detach().cpu()  # [3, H, W]

# 값 정규화 (시각화를 위해 0~1로)
original = torch.clamp(original, 0, 1)
filtered = torch.clamp(filtered, 0, 1)

# PIL 이미지로 변환
original_img = TF.to_pil_image(original)
filtered_img = TF.to_pil_image(filtered)

# 2. 시각화
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(filtered_img)
plt.title("Filtered Image (3x3 Kernel)")
plt.axis("off")

plt.tight_layout()
plt.show()
        