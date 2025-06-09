import cv2
import numpy as np
import os

# --- 参数设置 ---
# 图像路径
IMAGE_PATH = 'src/angle/cutResult/1748242778221_crop.jpg'  # 根据实际情况修改路径

# 阈值分割参数
THRESHOLD_VALUE = 20  # 灰度阈值

# Canny边缘检测参数
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# # 创建结果保存目录
# RESULT_DIR = 'src/angle/cannyResult'
# os.makedirs(RESULT_DIR, exist_ok=True)

# 1. 加载原始图像
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"错误：无法加载图片，请检查路径 '{IMAGE_PATH}' 是否正确。")
else:
    # 2. 图像预处理
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. 阈值分割
    # 应用二值化阈值处理（反转二值化 - THRESH_BINARY_INV）
    _, binary_img = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)
    
    # 保存二值化结果
    cv2.imwrite(os.path.join(RESULT_DIR, 'binary.png'), binary_img)
    
    # 4. 边缘检测
    # 应用Canny边缘检测
    edges = cv2.Canny(binary_img, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    
    # # 保存边缘检测结果
    # cv2.imwrite(os.path.join(RESULT_DIR, 'edges.png'), edges)
    
    # 显示处理结果
    cv2.imshow('Original Image', img)
    cv2.imshow('Binary Image', binary_img)
    cv2.imshow('Canny Edges', edges)
    
    # print(f"处理完成。结果已保存到 '{RESULT_DIR}' 目录。")
    
    # 等待用户按键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()

