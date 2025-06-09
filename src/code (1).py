import cv2
import numpy as np
import json
import os
import math

# ==============================================================================
# 模块一：YOLO 目标检测
# ==============================================================================
def detect_objects(image_path, model_path, class_names, conf_threshold=0.25, iou_threshold=0.45):
    """
    使用 YOLOv5 ONNX 模型检测图片中的目标。
    """
    net = cv2.dnn.readNetFromONNX(model_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return {os.path.basename(image_path): {}}

    h_orig, w_orig, _ = img.shape
    input_height, input_width = 640, 640

    scale = min(input_width / w_orig, input_height / h_orig)
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    padded_img = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    padded_img[(input_height - new_h) // 2 : (input_height - new_h) // 2 + new_h, \
               (input_width - new_w) // 2 : (input_width - new_w) // 2 + new_w, :] = cv2.resize(img, (new_w, new_h))

    blob = cv2.dnn.blobFromImage(padded_img, 1/255.0, (input_width, input_height), swapRB=True, crop=False)
    net.setInput(blob)
    output_data = net.forward()

    boxes, confidences, class_ids = [], [], []
    for detection in output_data[0]:
        final_scores = detection[4] * detection[5:]
        class_id = np.argmax(final_scores)
        confidence = final_scores[class_id]
        if confidence > conf_threshold:
            x_center, y_center, width_pred, height_pred = detection[:4]
            x_top_left = int(x_center - width_pred / 2)
            y_top_left = int(y_center - height_pred / 2)
            boxes.append([x_top_left, y_top_left, int(width_pred), int(height_pred)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
    
    detected_locks, detected_keys = [], []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            pad_w, pad_h = (input_width - new_w) / 2, (input_height - new_h) / 2
            final_x = int((x - pad_w) / scale)
            final_y = int((y - pad_h) / scale)
            final_w = int(w / scale)
            final_h = int(h / scale)
            final_x, final_y = max(0, final_x), max(0, final_y)
            final_w, final_h = min(w_orig - final_x, final_w), min(h_orig - final_y, final_h)
            
            detection_info = {"x": final_x, "y": final_y, "w": final_w, "h": final_h}
            if class_names[class_ids[i]] == 'lock':
                detected_locks.append(detection_info)
            elif class_names[class_ids[i]] == 'key':
                detected_keys.append(detection_info)

    results = {"x": 0, "y": 0, "w": 0, "h": 0, "is_key_in": False}
    if detected_locks:
        results.update(detected_locks[0])
    if detected_keys:
        results["is_key_in"] = True

    return {os.path.basename(image_path): results}


# ==============================================================================
# 模块二：距离计算
# ==============================================================================
def calculate_distance_geometric(detection_info):
    """
    使用对数公式计算锁孔到相机的距离。
    """
    if not isinstance(detection_info, dict) or 'w' not in detection_info or 'h' not in detection_info:
        return None
    w_pixel, h_pixel = detection_info.get('w'), detection_info.get('h')
    if w_pixel is None or h_pixel is None or w_pixel <= 0 or h_pixel <= 0:
        return None
    inv_sqrt_area = 1 / math.sqrt(w_pixel * h_pixel)
    distance_cm = 18.5910 * math.log(inv_sqrt_area) + 107.7396
    return max(0, distance_cm)


# ==============================================================================
# 模块三：角度识别 (核心整合部分 - 无可视化)
# ==============================================================================
def recognize_lock_angle(image_path, lock_bbox_info):
    """
    从原始图像中根据边界框裁剪出锁孔，并计算其旋转角度和归位状态。(无可视化版本)

    Args:
        image_path (str): 原始图片的完整路径。
        lock_bbox_info (dict): 包含锁孔 'x', 'y', 'w', 'h' 的字典。

    Returns:
        tuple: (angle, is_original) 或 (None, None) 如果失败。
    """
    # --- 内部辅助函数 ---
    def find_two_largest_peaks(data):
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append((i, data[i]))
        if len(peaks) < 2:
            max_idx = np.argmax(data)
            return [(max_idx, data[max_idx])]
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[:2]
    
    # --- 主流程开始 ---
    try:
        # 1. 从原图中裁剪锁孔区域
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"Angle Error: 无法读取原始图片 {image_path}")
            return None, None
            
        x, y, w, h = lock_bbox_info['x'], lock_bbox_info['y'], lock_bbox_info['w'], lock_bbox_info['h']
        cropped_img = original_img[y:y+h, x:x+w]

        # 2. 预处理裁剪后的图像
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        _, seg = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg, connectivity=8)

        if num_labels <= 1:
            print("Angle Error: 预处理后未找到锁孔主体。")
            return None, None

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_component_label = np.argmax(areas) + 1
        binary_image = np.zeros_like(seg)
        binary_image[labels == largest_component_label] = 255
        
        # 3. 计算角度
        h_bin, w_bin = binary_image.shape
        moments = cv2.moments(binary_image)
        if moments["m00"] == 0:
            print("Angle Error: 锁孔二值图为空。")
            return None, None
            
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        center_point = (center_x, center_y)

        projections = []
        for angle_rot in range(180):
            M = cv2.getRotationMatrix2D(center_point, angle_rot, 1.0)
            rotated_image = cv2.warpAffine(binary_image, M, (w_bin, h_bin))
            projection_value = np.sum(rotated_image[:, center_x])
            projections.append(projection_value)
            
        top_two_peaks = find_two_largest_peaks(np.array(projections))
        peak1_angle = top_two_peaks[0][0]
        peak2_angle = top_two_peaks[1][0] if len(top_two_peaks) > 1 else peak1_angle

        if abs(peak1_angle - peak2_angle) > 90:
            bisector_angle = ((peak1_angle + peak2_angle + 180) / 2) % 180
        else:
            bisector_angle = (peak1_angle + peak2_angle) / 2

        final_axis_angle = 90 - bisector_angle
        keyhole_orientation = 90 - final_axis_angle
        if keyhole_orientation > 90: keyhole_orientation -= 180
            
        final_angle = -keyhole_orientation
        is_original = abs(final_angle) < 5.0

        return round(final_angle, 2), is_original

    except Exception as e:
        print(f"Angle Error: 在角度识别过程中发生未知错误: {e}")
        return None, None


# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    # --- 配置 ---
    model_path = "src/best.onnx"
    # 可以处理单个图片或一个图片列表
    image_paths = [
        "DataAll/1748242655899.jpg"
    ]
    class_names = ['lock', 'key']
    
    # --- 运行 ---
    all_results = {}
    for img_path in image_paths:
        print(f"\n===== 处理图片: {img_path} =====")
        
        # 1. 目标检测
        detection_result_dict = detect_objects(img_path, model_path, class_names)
        filename = os.path.basename(img_path)
        detection_info = detection_result_dict.get(filename, {})
        
        if not detection_info or detection_info.get("w", 0) <= 0:
            print("未检测到锁孔，跳过后续处理。")
            all_results.update(detection_result_dict)
            continue

        # 2. 计算距离
        distance = calculate_distance_geometric(detection_info)
        if distance is not None:
            detection_info["distance"] = round(distance, 2)
        
        # 3. 识别角度
        print("开始识别锁孔角度...")
        lock_angle, is_lock_original = recognize_lock_angle(img_path, detection_info)
        if lock_angle is not None:
            detection_info["lock_angle"] = lock_angle
            detection_info["is_lock_original"] = is_lock_original
            print(f"角度识别完成: Angle={lock_angle}, Is Original={is_lock_original}")
        else:
            print("角度识别失败。")
        
        all_results[filename] = detection_info

    # 打印最终的 JSON 格式结果
    print("\n\n===== 最终 JSON 输出 =====")
    print(json.dumps(all_results, indent=4))