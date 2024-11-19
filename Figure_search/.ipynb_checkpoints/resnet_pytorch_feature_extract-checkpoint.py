import os
import cv2
import pickle
import numpy as np
import torch
from torchvision import models, transforms
import torch.nn as nn

# 设置设备为 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图像尺寸
w, h = 224, 224

# 加载预训练的 ResNet50 模型，不包括全连接层
# original_model = models.resnet50(pretrained=True)

original_model = models.resnet18(pretrained=False)
original_model.fc = nn.Linear(in_features=512, out_features=13, bias=True)

# Load your local weights into the model
weight_path = '/mnt/workspace/pytorch_图片分类/models/best.pth'  # Replace with your actual weight file path
original_model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)

# 创建一个新的模型，输出平均池化层之前的特征
class ResNet18Embeddings(torch.nn.Module):
    def __init__(self, original_model):
        super(ResNet18Embeddings, self).__init__()
        # 使用除 avgpool 和 fc 以外的所有层
        self.features = torch.nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            original_model.layer4
        )

    def forward(self, x):
        x = self.features(x)
        return x

# 初始化嵌入模型
model = ResNet18Embeddings(original_model)
model.eval()  # 设置模型为评估模式

# 将模型移动到 GPU（如果可用）
model.to(device)


preprocess = transforms.Compose([
                #transforms.Lambda(source.convert_img),
                #transforms.Grayscale(num_output_channels=3),
                #transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(1, 1)),# 转换为灰度图
                transforms.ToTensor(),

                transforms.Resize(size=(224, 224), antialias=True),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
# 数据集的基路径
base_path = r"/mnt/workspace/datasets"

# 获取所有图片路径
files = [os.path.join(base_path, file) for file in os.listdir(base_path)]

# 用于存储嵌入的列表
embeddings = []

for file in files:
    # 使用 OpenCV 读取图像
    source = cv2.imread(file)
    if not isinstance(source, np.ndarray):
        continue
    # 将 BGR 转换为 RGB 格式
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    # 应用预处理转换
    input_tensor = preprocess(source)
    # 添加批次维度
    input_batch = input_tensor.unsqueeze(0)  # 形状：[1, 3, 224, 224]

    # 将输入张量移动到 GPU
    input_batch = input_batch.to(device)

    # 在不计算梯度的情况下获取嵌入
    with torch.no_grad():
        output = model(input_batch)

    # 将输出移动到 CPU 并重塑为一维向量
    embedding = output.cpu().view(-1).numpy()

    # 将嵌入和文件路径添加到列表
    embeddings.append({
        "filepath": file,
        "embedding": embedding
    })

# 将嵌入保存到一个 pickle 文件
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
