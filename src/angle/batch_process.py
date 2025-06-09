import cv2
import numpy as np
import os
import glob

def process_image(image_path, output_dir):
    """
    处理单张图片并保存结果
    
    参数:
        image_path: 输入图片路径
        output_dir: 输出目录路径
    """
    # 1. 读取图片
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法找到或打开图片文件 '{image_path}'")
    except Exception as e:
        print(f"处理 {image_path} 时出错: {e}")
        return False
    
    # 2. 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. 应用阈值分割
    _, seg = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    
    # 4. 使用连通组件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg, connectivity=8, ltype=cv2.CV_32S)
    
    # 5. 找到最大连通组件
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]  # 提取所有前景组件的面积
        largest_component_label = np.argmax(areas) + 1
        
        # 创建只包含最大组件的图像
        output_img = np.zeros_like(seg)
        output_img[labels == largest_component_label] = 255
    else:
        # 如果没有前景对象，输出就是全黑图像
        output_img = np.zeros_like(seg)
    
    # 6. 保存结果
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, output_img)
    
    return True

def main():
    # 定义输入和输出路径
    input_dir = 'src/angle/cutResult/keyIn'
    output_dir = 'src/angle/cutResult/keyIn/result'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图片文件
    # 支持常见图片格式
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_paths = []
    
    for pattern in image_patterns:
        image_paths.extend(glob.glob(os.path.join(input_dir, pattern)))
    
    if not image_paths:
        print(f"在 {input_dir} 目录下未找到任何图片文件")
        return
    
    # 处理每张图片
    successful = 0
    failed = 0
    
    print(f"开始处理 {len(image_paths)} 张图片...")
    for i, image_path in enumerate(image_paths):
        print(f"处理图片 {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        if process_image(image_path, output_dir):
            successful += 1
        else:
            failed += 1
    
    print(f"\n处理完成!")
    print(f"成功处理: {successful} 张图片")
    print(f"处理失败: {failed} 张图片")
    print(f"结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()