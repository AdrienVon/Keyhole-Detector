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
    # 忽略边界，只在中间寻找
    for i in range(1, len(data) - 1):
        if data[i] > data[i-1] and data[i] > data[i+1]:
            peaks.append((i, data[i])) # 存储 (角度, 峰值高度)

    if len(peaks) < 2:
        # 如果找不到足够的峰值，就返回全局最大值作为唯一的峰值
        max_idx = np.argmax(data)
        return [(max_idx, data[max_idx])]

    # 按峰值高度降序排序，取前两个
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[:2]


def analyze_with_polar_transform(image_path):
    """
    终极版(高效法)：使用极坐标变换生成投影，寻找双峰并计算其锐角角平分线。
    """
    # --- 步骤 0: 加载图像 ---
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None: return
    
    # 确保图像是干净的二值图
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # --- 步骤 1: 寻找中心点 ---
    height, width = image.shape
    moments = cv2.moments(image)
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    center_point = (center_x, center_y)

    # --- 步骤 2: 核心！使用极坐标变换生成投影数据 ---
    print("正在使用极坐标变换生成投影数据...")
    # 计算最大半径以覆盖整个图像
    max_radius = int(math.sqrt((height/2)**2 + (width/2)**2))
    
    # 执行一次性的极坐标变换
    polar_image = cv2.warpPolar(image, (360, max_radius), center_point, max_radius, cv2.WARP_FILL_OUTLIERS)
    
    # 创建投影数组
    line_projection = np.zeros(180, dtype=np.float32)
    for angle in range(180):
        # 一条穿过中心的直线，对应于极坐标图上相隔180度的两条垂直线
        col1 = polar_image[:, angle]
        col2 = polar_image[:, angle + 180]
        # 将两条线上（两列）的白色像素相加
        line_projection[angle] = np.sum(col1) + np.sum(col2)
    print("投影数据生成完毕。")

    # --- 步骤 3: 寻找两个最显著的峰值 ---
    top_two_peaks = find_two_largest_peaks(np.array(line_projection))
    
    if len(top_two_peaks) >= 2:
        # 峰值角度直接对应于原始图像中与水平轴的夹角
        peak1_angle = top_two_peaks[0][0]
        peak2_angle = top_two_peaks[1][0]
    else: # 鲁棒性后备
        peak1_angle = peak2_angle = top_two_peaks[0][0]
        
    # --- 步骤 4: 计算【锐角】角平分线 ---
    # 这里的角度就是相对于水平轴的角度
    if abs(peak1_angle - peak2_angle) > 90:
        final_axis_angle = ((peak1_angle + peak2_angle + 180) / 2) % 180
    else:
        final_axis_angle = (peak1_angle + peak2_angle) / 2

    # 锁孔本身的朝向与水平对称轴（角平分线）垂直
    keyhole_orientation = 90 - final_axis_angle
    if keyhole_orientation > 90: # 规范化到 -90 ~ 90 度
        keyhole_orientation -= 180
    
    print("\n--- 最终精确结果 (极坐标法) ---")
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
    # OpenCV的Y轴向下，sin前面用负号来校正到标准数学坐标系
    p1 = (int(center_x - line_len * math.cos(rad_bisector)), int(center_y - line_len * math.sin(rad_bisector)))
    p2 = (int(center_x + line_len * math.cos(rad_bisector)), int(center_y + line_len * math.sin(rad_bisector)))
    cv2.line(final_vis, p1, p2, (0, 0, 255), 2)
    
    cv2.imshow("Final Angle Detection (Polar Method)", final_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_file = 'src/angle/cannyResult/binary/1748242778221_crop.png' 
    analyze_with_polar_transform(image_file)