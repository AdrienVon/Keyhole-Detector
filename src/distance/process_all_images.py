import cv2
import numpy as np
import json
import os
import csv
import glob
from tqdm import tqdm  # 添加处理进度条

# 导入检测和距离计算函数
from NN4 import detect_objects, calculate_distance_geometric

def process_all_images_to_csv(data_dir, model_path, output_csv, class_names=None):
    """
    处理指定目录中的所有图片，并将检测结果输出到CSV文件。
    
    Args:
        data_dir (str): 包含图片的目录路径
        model_path (str): 模型文件的路径
        output_csv (str): 输出CSV文件的路径
        class_names (list): 类别名称列表
    """
    if class_names is None:
        class_names = ['lock', 'key']
    
    # 获取所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(data_dir, ext)))
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 准备CSV文件
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['imgname', 'x', 'y', 'w', 'h', 'distance', 'is_key_in']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 处理每张图片
        for img_path in tqdm(image_files):
            # 进行目标检测
            result = detect_objects(img_path, model_path, class_names)
            
            # 获取文件名
            filename = os.path.basename(img_path)
            
            # 如果检测到了锁孔，尝试计算距离
            if filename in result and result[filename] and result[filename]["w"] > 0:
                # 计算距离
                detection_info = result[filename]
                distance = calculate_distance_geometric(detection_info)
                
                # 写入CSV
                csv_row = {
                    'imgname': filename,
                    'x': detection_info['x'],
                    'y': detection_info['y'],
                    'w': detection_info['w'],
                    'h': detection_info['h'],
                    'distance': round(distance, 2) if distance else '',
                    'is_key_in': detection_info['is_key_in']
                }
                writer.writerow(csv_row)
            else:
                # 没有检测到锁孔，写入空行
                csv_row = {
                    'imgname': filename,
                    'x': '',
                    'y': '',
                    'w': '',
                    'h': '',
                    'distance': '',
                    'is_key_in': ''
                }
                writer.writerow(csv_row)
    
    print(f"处理完成，结果已保存到 {output_csv}")

if __name__ == "__main__":
    # 配置路径
    data_dir = "DataAll"  # 图片目录
    model_path = "src/best.onnx"  # 模型文件路径
    output_csv = "detection_results.csv"  # 输出CSV文件路径
    class_names = ['lock', 'key']  # 类别名称
    
    # 确保数据目录存在
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        # 也可以尝试从当前目录相对路径查找
        data_dir = os.path.join(os.path.dirname(__file__), "..", "DataAll")
        if os.path.exists(data_dir):
            print(f"找到数据目录: {data_dir}")
        else:
            print("请确保 DataAll 目录存在，或者提供正确的路径")
            exit(1)
    
    # 处理所有图片并输出到CSV
    process_all_images_to_csv(data_dir, model_path, output_csv, class_names)