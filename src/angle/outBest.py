import cv2
import numpy as np
import math

# ==============================================================================
# 模块一：图像预处理
# ==============================================================================
def preprocess_image(image_path):
    """
    加载原始图像，进行阈值分割，并提取最大的连通组件。
    
    参数:
        image_path (str): 原始锁孔图像的文件路径。
        
    返回:
        numpy.ndarray: 只包含最大组件的干净二值图，如果出错则返回 None。
    """
    print(f"--- 步骤 A: 正在预处理图像 '{image_path}' ---")
    try:
        # 1. 读取原始图像
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法找到或打开图片文件: {image_path}")

        # 2. 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. 应用反向二值阈值分割
        _, seg = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)

        # 4. 使用连通组件分析来去噪
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg, connectivity=8)

        if num_labels > 1:
            # 5. 找到最大连通组件（排除背景）
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_component_label = np.argmax(areas) + 1

            # 6. 创建一个只包含最大组件的输出图像
            output_img = np.zeros_like(seg)
            output_img[labels == largest_component_label] = 255
            
            print("预处理成功：已分离出最大组件。")
            # 可选：显示预处理结果
            # cv2.imshow('Preprocessed Image', output_img)
            # cv2.waitKey(1)
            
            return output_img
        else:
            print("警告：在预处理中未找到任何前景对象。")
            return None
            
    except Exception as e:
        print(f"预处理时发生错误: {e}")
        return None

# ==============================================================================
# 模块二：角度计算
# ==============================================================================
def find_two_largest_peaks(data):
    """一个简单的函数，用于在1D数据中找到两个最大的局部峰值"""
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            peaks.append((i, data[i]))
    if len(peaks) < 2:
        max_idx = np.argmax(data)
        return [(max_idx, data[max_idx])]
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[:2]

def add_text_to_image(image, text, org=(10, 30), color=(0, 255, 0)):
    """在图像上添加说明文字的辅助函数"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    thickness = 2
    cv2.putText(image, text, org, font, fontScale, (0, 0, 0), thickness + 3)
    cv2.putText(image, text, org, font, fontScale, color, thickness)
    return image

def calculate_keyhole_angle(binary_image):
    """
    接收一张干净的二值图，计算其旋转角度。
    
    参数:
        binary_image (numpy.ndarray): 预处理后的二值图。
        
    返回:
        float: 计算出的旋转角度，如果出错则返回 None。
    """
    print("\n--- 步骤 B: 开始计算角度 ---")
    # 确保传入的是二值图
    if binary_image is None or len(binary_image.shape) > 2:
        print("错误：传入角度计算的图像格式不正确。")
        return None

    # --- 步骤 B.1: 寻找中心点 ---
    height, width = binary_image.shape
    moments = cv2.moments(binary_image)
    if moments["m00"] == 0:
        print("错误：传入的二值图为空，无法计算中心点。")
        return None
        
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    center_point = (center_x, center_y)

    # --- 步骤 B.2: 循环180次生成投影数据 ---
    print("正在生成投影数据 (循环180次)...")
    projections = []
    for angle in range(180):
        M = cv2.getRotationMatrix2D(center_point, angle, 1.0)
        rotated_image = cv2.warpAffine(binary_image, M, (width, height))
        projection_value = np.sum(rotated_image[:, center_x])
        projections.append(projection_value)
    print("投影数据生成完毕。")
    
    # --- 步骤 B.3: 寻找两个最显著的峰值 ---
    top_two_peaks = find_two_largest_peaks(np.array(projections))
    
    if len(top_two_peaks) >= 2:
        peak1_angle, peak2_angle = top_two_peaks[0][0], top_two_peaks[1][0]
    else:
        peak1_angle = peak2_angle = top_two_peaks[0][0]
        
    # --- 步骤 B.4: 计算【锐角】角平分线 ---
    if abs(peak1_angle - peak2_angle) > 90:
        bisector_angle = ((peak1_angle + peak2_angle + 180) / 2) % 180
    else:
        bisector_angle = (peak1_angle + peak2_angle) / 2

    final_axis_angle = 90 - bisector_angle
    
    keyhole_orientation = 90 - final_axis_angle
    if keyhole_orientation > 90:
        keyhole_orientation -= 180
        
    final_angle = -keyhole_orientation

    print("\n--- 角度计算完成 ---")
    print(f"检测到的两个主峰角度: {peak1_angle}° 和 {peak2_angle}°")
    print(f"计算出的水平对称轴 (锐角角平分线) 角度: {final_axis_angle:.2f}°")

    # --- 最终可视化 ---
    color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    text = f"Keyhole Angle: {final_angle:.2f} deg"
    final_vis = add_text_to_image(color_image.copy(), text)
    add_text_to_image(final_vis, f"Symmetry Axis: {final_axis_angle:.1f} deg", org=(10, 60), color=(0, 0, 255))
    
    rad_bisector = math.radians(final_axis_angle)
    line_len = int(width)
    p1 = (int(center_x - line_len * math.cos(rad_bisector)), int(center_y - line_len * math.sin(rad_bisector)))
    p2 = (int(center_x + line_len * math.cos(rad_bisector)), int(center_y + line_len * math.sin(rad_bisector)))
    cv2.line(final_vis, p1, p2, (0, 0, 255), 2)
    
    cv2.imshow("Final Result", final_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return final_angle

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    # ------------------- 输入区 -------------------
    # 在这里修改你要分析的原始锁孔图片路径
    input_image_path = 'src/angle/cutResult/keyOut/1748242785054_crop.jpg'
    # ----------------------------------------------

    # 执行预处理
    clean_binary_image = preprocess_image(input_image_path)
    
    # 如果预处理成功，则执行角度计算
    if clean_binary_image is not None:
        final_rotation_angle = calculate_keyhole_angle(clean_binary_image)
        
        if final_rotation_angle is not None:
            print("\n=======================================================")
            print(f">>> 最终结果: 锁孔相对于垂直方向的旋转角度为 {final_rotation_angle:.2f} 度")
            print("=======================================================")