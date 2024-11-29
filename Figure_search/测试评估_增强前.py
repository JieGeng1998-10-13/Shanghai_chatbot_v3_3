import os
import cv2
import numpy as np
from tqdm import tqdm  # 用于显示进度条
from app_figure_search_before import search_image


def get_label_prefix(s):
    return s.split(' ')[0]


def evaluate_system(dataset_folder, K=10):
    """
    评估图像检索系统的性能。

    参数：
    - dataset_folder: 存放测试图片的文件夹路径
    - K: 评估指标中的前K个检索结果

    返回：
    - 包含评估指标的字典
    """
    # 存储每个查询的评估指标
    all_precisions = []
    all_average_precisions = []
    all_recalls = []

    # 获取测试集中的所有图片文件
    image_files = [f for f in os.listdir(dataset_folder) if f.endswith('.jpg')]

    # 遍历每张测试图片
    for image_file in tqdm(image_files, desc="Evaluating"):
        # 构建完整的图片路径
        image_path = os.path.join(dataset_folder, image_file)
        
        # 读取图片
        image = cv2.imread(image_path)
        #print("正在查询图片")
        if image is None:
            print(f"无法读取图片：{image_path}")
            continue
        
        # 将图片转换为 RGB 格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 从文件名中获取真实标签（去掉 '.jpg' 后缀）
        ground_truth_label = os.path.splitext(image_file)[0]
        
        # 使用现有的检索函数获取检索结果
        output_images, output_paths_str = search_image(image)
        
        # 从检索结果中提取标签
        # 假设 output_paths_str 的格式为：
        # "第1种可能的产品型号为： label1\n第2种可能的产品型号为： label2\n..."
        retrieved_labels = []
        lines = output_paths_str.strip().split('\n')
        for line in lines[:K]:  # 只考虑前 K 个结果
            # 提取每行中的标签
            parts = line.split('：')
            if len(parts) == 2:
                label = parts[1].strip()
                retrieved_labels.append(label)
            else:
                print(f"无法解析行：{line}")
        
        # 判断每个检索结果是否与真实标签匹配
        relevant_recall = [1 if label == ground_truth_label else 0 for label in retrieved_labels]
        relevant_pre = [
            1 if get_label_prefix(label) == get_label_prefix(ground_truth_label) else 0
            for label in retrieved_labels
        ]

        # 计算 Precision@K
        precision_at_k = sum(relevant_pre) / K
        all_precisions.append(precision_at_k)
        
        # 计算 Recall@K
        # 假设每个查询在数据库中只有一个相关项
        total_relevant = 1  # 如果有多个相关项，请相应调整
        recall_at_k = sum(relevant_recall) / total_relevant
        all_recalls.append(recall_at_k)
        
        # 计算 Average Precision (AP)
        num_relevant = 0
        precisions = []
        for idx, rel in enumerate(relevant_recall):
            if rel:
                num_relevant += 1
                precisions.append(num_relevant / (idx + 1))
        if precisions:
            average_precision = sum(precisions) / total_relevant
        else:
            average_precision = 0.0
        all_average_precisions.append(average_precision)
    
    # 计算平均指标
    mean_precision_at_k = np.mean(all_precisions)
    mean_recall_at_k = np.mean(all_recalls)
    mAP = np.mean(all_average_precisions)
    
    print(f"平均 Precision@{K}: {mean_precision_at_k:.4f}")
    print(f"平均 Recall@{K}: {mean_recall_at_k:.4f}")
    print(f"平均平均精度 (mAP): {mAP:.4f}")
    
    return {
        'mean_precision_at_k': mean_precision_at_k,
        'mean_recall_at_k': mean_recall_at_k,
        'mean_average_precision': mAP
    }

# 示例用法：
if __name__ == "__main__":
    dataset_folder = '/mnt/上海市交通系统交易问答框架/Figure_search/datatest'  # 将此处替换为您的 dataset 文件夹路径
    evaluate_system(dataset_folder, K=10)