import cv2
import numpy as np
import math

def add_text_to_image(image, text, org=(10, 30), color=(0, 255, 0)):
    """在图像上添加说明文字的辅助函数"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    thickness = 2
    cv2.putText(image, text, org, font, fontScale, (0, 0, 0), thickness + 3)
    cv2.putText(image, text, org, font, fontScale, color, thickness)
    return image

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

def analyze_with_rotation_loop(image_path):
    """
    最终确认版(循环法)：通过寻找投影双峰，并计算其【锐角角平分线】，来确定最终角度。
    """
    # --- 步骤 0: 加载图像 ---
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None: return
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # --- 步骤 1: 寻找中心点 ---
    height, width = image.shape
    moments = cv2.moments(image)
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    center_point = (center_x, center_y)

    # --- 步骤 2: 核心！循环180次生成投影数据 ---
    print("正在生成投影数据 (循环180次)...")
    projections = []
    for angle in range(180):
        M = cv2.getRotationMatrix2D(center_point, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (width, height))
        projection_value = np.sum(rotated_image[:, center_x])
        projections.append(projection_value)
    print("投影数据生成完毕。")
    
    # --- 步骤 3: 寻找两个最显著的峰值 ---
    top_two_peaks = find_two_largest_peaks(np.array(projections))
    
    if len(top_two_peaks) >= 2:
        peak1_angle = top_two_peaks[0][0]
        peak2_angle = top_two_peaks[1][0]
    else: # 鲁棒性后备
        peak1_angle = peak2_angle = top_two_peaks[0][0]
        
    # --- 步骤 4: 计算【锐角】角平分线 ---
    if abs(peak1_angle - peak2_angle) > 90:
        # 两个峰值的直接角度差是钝角，说明锐角跨越了0/180度边界。
        # 此计算方法能正确找到跨越边界的锐角角平分线。
        bisector_angle = ((peak1_angle + peak2_angle + 180) / 2) % 180
    else:
        # 两个峰值的直接角度差是锐角，直接取平均值即可。
        bisector_angle = (peak1_angle + peak2_angle) / 2

    # 转换：我们找到的角度是“图像旋转多少度，对称轴能垂直”
    # 所以对称轴在原始图像中与水平线的夹角是 90 - bisector_angle
    final_axis_angle = 90 - bisector_angle
    
    # 锁孔本身的朝向与水平对称轴（角平分线）垂直
    keyhole_orientation = 90 - final_axis_angle
    if keyhole_orientation > 90: # 规范化到 -90 ~ 90 度
        keyhole_orientation -= 180
    
    print("\n--- 最终精确结果 ---")
    print(f"检测到的两个主峰角度: {peak1_angle}° 和 {peak2_angle}°")
    print(f"计算出的水平对称轴 (锐角角平分线) 角度: {final_axis_angle:.2f}°")
    print(f"锁孔本身相对于垂直方向的旋转角度: {-keyhole_orientation:.2f}°")

    # --- 最终可视化 ---
    final_vis = color_image.copy()
    text = f"Keyhole Angle: {-keyhole_orientation:.2f} deg"
    final_vis = add_text_to_image(final_vis, text)
    add_text_to_image(final_vis, f"Symmetry Axis: {final_axis_angle:.1f} deg", org=(10, 60), color=(0, 0, 255))
    
    # 绘制水平对称轴 (红色)
    rad_bisector = math.radians(final_axis_angle)
    line_len = int(width / 2)
    p1 = (int(center_x - line_len * math.cos(rad_bisector)), int(center_y - line_len * math.sin(rad_bisector)))
    p2 = (int(center_x + line_len * math.cos(rad_bisector)), int(center_y + line_len * math.sin(rad_bisector)))
    cv2.line(final_vis, p1, p2, (0, 0, 255), 2)
    
    cv2.imshow("Final Angle Detection (Acute Bisector)", final_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_file = 'src/angle/cannyResult/binary/1748242778221_crop.png' 
    analyze_with_rotation_loop(image_file)