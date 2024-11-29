
import os
import cv2
import joblib
import random
import numpy as np
import torch
import gradio as gr
from torchvision import models, transforms
from PIL import Image  # 确保导入 PIL.Image
from pymilvus import connections, Collection
import torch.nn as nn


# 设置设备为 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 PCA 模型和 ResNet50 编码器
pca = joblib.load('/mnt/上海市交通系统交易问答框架/Figure_search/pca.pkl')
w, h = 224, 224

# 加载预训练的 ResNet50 模型，不包括全连接层
original_model = models.resnet18(pretrained=False)
original_model.fc = nn.Linear(in_features=512, out_features=13, bias=True)

# Load your local weights into the model
weight_path = '/mnt/上海市交通系统交易问答框架/Figure_search/best.pth'  # Replace with your actual weight file path
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

# 初始化编码器模型
encoder = ResNet18Embeddings(original_model)
encoder.eval()  # 设置模型为评估模式
encoder.to(device)  # 将模型移动到 GPU（如果可用）

# 定义预处理转换（移除了 transforms.ToPILImage()）


preprocess = transforms.Compose([
                #transforms.Lambda(convert_img),
                #transforms.Grayscale(num_output_channels=3),
                #transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(1, 1)),# 转换为灰度图
                transforms.ToTensor(),

                transforms.Resize(size=(224, 224), antialias=True),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
# 连接到 Milvus 数据库
connections.connect(host='47.102.103.246', port='19530')
collection = Collection(name='images_final')
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
collection.load()

# 图像处理函数
def process_image(image):
    # image 是一个 RGB 格式的 numpy 数组
    # 将 numpy 数组转换为 PIL 图像
    image_pil = Image.fromarray(image)
    # 应用预处理转换
    input_tensor = preprocess(image_pil)
    # 添加批次维度并移动到设备
    input_batch = input_tensor.unsqueeze(0).to(device)
    # 在不计算梯度的情况下获取嵌入
    with torch.no_grad():
        embedding = encoder(input_batch)
    # 将输出重塑为一维向量，转换为 numpy 数组，并移动到 CPU
    embedding = embedding.view(1, -1).cpu().numpy()
    # 应用 PCA 转换
    return pca.transform(embedding)

# 检索并返回结果（每个结果包含图片和路径）
def search_image(uploaded_image):
    # 获取上传图片的特征
    target_embedding = process_image(uploaded_image)

    # 在数据库中进行检索
    results = collection.search(
        data=[target_embedding[0]],
        anns_field='embedding',
        param=search_params,
        output_fields=['filepath'],
        limit=10,
        consistency_level="Strong"
    )

    # 获取检索结果的图像和文件路径
    output_images = []
    output_paths = []
    for result in results[0]:
        filepath = result.entity.get('filepath')
        image = cv2.imread(filepath)
        if image is None:
            continue
        # 转换为 RGB 格式并调整大小
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (w, h))
        output_images.append(image)
        output_paths.append(filepath)  # 保存文件路径

    # 将文件路径列表转换为字符串
    #output_paths_str = "\n".join(output_paths)
    formatted_paths = [
        f"第{i+1}种可能的产品型号为： {os.path.splitext(os.path.basename(path))[0]}"  # 提取文件名并格式化
        for i, path in enumerate(output_paths)
    ]
    return output_images, "\n".join(formatted_paths)



# Gradio 界面
iface = gr.Interface(
    fn=search_image, 
    inputs=gr.Image(type="numpy"), 
    outputs=[gr.Gallery(label="Similar Images"), gr.Textbox(label="File Paths",lines=10)],
    live=True, 
    title="上海市交通交易系统商品采购搜索",
    description="上传一张商品图片进行检索，并返回最相似的商品图片和对应的商品型号"
)

if __name__ == "__main__":
    # 启动 Gradio 前端界面
    iface.launch(share=True)
