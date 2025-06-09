import os
import cv2
import json
import glob
from tqdm import tqdm
import sys
import os

# 添加父目录到系统路径，以便导入NN4模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NN4 import detect_objects

def process_and_crop_images(input_dir, output_dir, model_path, class_names, padding=10):
    """
    处理指定目录中的所有图片，检测锁孔并裁剪，保存到输出目录
    检测到钥匙的图片保存到keyIn子目录，未检测到钥匙的保存到keyOut子目录
    
    Args:
        input_dir (str): 输入图片目录
        output_dir (str): 输出裁剪图片目录
        model_path (str): 模型路径
        class_names (list): 类别名称列表
        padding (int): 裁剪边界框时额外添加的边距(像素)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建keyIn和keyOut子目录
    key_in_dir = os.path.join(output_dir, "keyIn")
    key_out_dir = os.path.join(output_dir, "keyOut")
    os.makedirs(key_in_dir, exist_ok=True)
    os.makedirs(key_out_dir, exist_ok=True)
    
    # 获取所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"在 {input_dir} 中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件，开始处理...")
    
    # 统计计数器
    key_in_count = 0
    key_out_count = 0
    no_keyhole_count = 0
    
    # 处理每张图片
    for img_path in tqdm(image_files):
        # 调用检测函数
        result = detect_objects(img_path, model_path, class_names)
        
        # 获取文件名
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        # 检查是否检测到锁孔
        if filename in result and result[filename] and result[filename]["w"] > 0:
            # 获取原始图片
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图片: {img_path}，跳过")
                continue
                
            # 获取锁孔边界框
            bbox = result[filename]
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            
            # 添加边距，但确保不超出图像边界
            h_img, w_img = img.shape[:2]
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_img, x + w + padding)
            y2 = min(h_img, y + h + padding)
            
            # 裁剪图像
            cropped_img = img[y1:y2, x1:x2]
            
            # 根据是否有钥匙决定保存到哪个文件夹
            if result[filename]["is_key_in"] == True:
                output_path = os.path.join(key_in_dir, f"{base_name}_crop.jpg")
                cv2.imwrite(output_path, cropped_img)
                key_in_count += 1
            else:
                output_path = os.path.join(key_out_dir, f"{base_name}_crop.jpg")
                cv2.imwrite(output_path, cropped_img)
                key_out_count += 1
        else:
            # 如果没有检测到锁孔，记录但不保存
            no_keyhole_count += 1
    
    print(f"处理完成!")
    print(f"总图片数: {len(image_files)}")
    print(f"包含钥匙的锁孔图片 (保存到 {key_in_dir}): {key_in_count}")
    print(f"不包含钥匙的锁孔图片 (保存到 {key_out_dir}): {key_out_count}")
    print(f"未检测到锁孔的图片: {no_keyhole_count}")
    print(f"结果保存在: {output_dir}")

if __name__ == "__main__":
    # 路径配置
    input_directory = "src/angle/images"
    output_directory = "src/angle/cutResult"
    model_path = "src/best.onnx"
    class_names = ['lock', 'key']
    
    # 检查输入目录是否存在
    if not os.path.exists(input_directory):
        print(f"错误: 输入目录 {input_directory} 不存在")
        exit(1)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        exit(1)
    
    # 处理图片并裁剪
    process_and_crop_images(input_directory, output_directory, model_path, class_names)