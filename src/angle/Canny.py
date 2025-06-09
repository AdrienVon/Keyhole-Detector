import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

# --- 参数设置 ---
# 目录路径
INPUT_DIR = 'src/angle/cutResult'  # 输入裁剪后的图片目录
OUTPUT_DIR = 'src/angle/cannyResult'  # 输出基本目录

# 创建三个子目录分别存放不同类型的结果
BINARY_DIR = os.path.join(OUTPUT_DIR, 'binary')  # 二值化结果目录
EDGES_DIR = os.path.join(OUTPUT_DIR, 'edges')    # 边缘检测结果目录
COMPOSITE_DIR = os.path.join(OUTPUT_DIR, 'composite')  # 组合图像结果目录

# 阈值分割参数
THRESHOLD_VALUE = 20  # 灰度阈值

# Canny边缘检测参数
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

def process_image(img_path):
    """
    处理单张图片，进行边缘检测并保存不同结果到各自文件夹
    
    Args:
        img_path (str): 输入图片路径
    
    Returns:
        bool: 处理是否成功
    """
    # 获取文件名（不含路径和扩展名）
    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]
    
    # 加载原始图像
    img = cv2.imread(img_path)
    if img is None:
        return False
        
    # 图像预处理 - 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 阈值分割 - 应用二值化阈值处理（反转二值化）
    _, binary_img = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)
    
    # 保存二值化结果到binary文件夹
    binary_path = os.path.join(BINARY_DIR, f'{base_name}.png')
    cv2.imwrite(binary_path, binary_img)
    
    # 边缘检测 - 应用Canny边缘检测
    edges = cv2.Canny(binary_img, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    
    # 保存边缘检测结果到edges文件夹
    edges_path = os.path.join(EDGES_DIR, f'{base_name}.png')
    cv2.imwrite(edges_path, edges)
    
    # 保存带有边缘的原图到composite文件夹
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    img_with_edges = cv2.addWeighted(img, 0.8, edges_color, 0.2, 0)
    composite_path = os.path.join(COMPOSITE_DIR, f'{base_name}.png')
    cv2.imwrite(composite_path, img_with_edges)
    
    return True

def main():
    # 创建各输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BINARY_DIR, exist_ok=True)
    os.makedirs(EDGES_DIR, exist_ok=True)
    os.makedirs(COMPOSITE_DIR, exist_ok=True)
    
    # 获取所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext.upper())))
    
    if not image_files:
        print(f"在 {INPUT_DIR} 中没有找到图片文件")
        return
    
    print(f"开始处理 {len(image_files)} 个图片...")
    
    # 处理每张图片
    success_count = 0
    failed_count = 0
    
    for img_path in tqdm(image_files):
        if process_image(img_path):
            success_count += 1
        else:
            failed_count += 1
            print(f"无法处理图片: {img_path}")
    
    print(f"\n处理完成:")
    print(f"- 成功处理: {success_count} 张图片")
    print(f"- 处理失败: {failed_count} 张图片")
    print(f"结果已保存到以下目录:")
    print(f"- 二值化图像: {BINARY_DIR}")
    print(f"- 边缘检测图像: {EDGES_DIR}")
    print(f"- 组合效果图像: {COMPOSITE_DIR}")

if __name__ == "__main__":
    main()