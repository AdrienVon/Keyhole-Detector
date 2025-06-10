import cv2
import numpy as np
import json
import os
import math
import time

# ==============================================================================
# 模块一：YOLO 目标检测
# ==============================================================================
def detect_objects(img, net, class_names, conf_threshold=0.25, iou_threshold=0.45):
    # (此函数无需修改，保持原样)
    if img is None: return {}
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
            boxes.append([int(x_center - width_pred / 2), int(y_center - height_pred / 2), int(width_pred), int(height_pred)])
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
    return results

# ==============================================================================
# 模块二：距离计算
# ==============================================================================
def calculate_distance_geometric(detection_info):
    # (此函数无需修改，保持原样)
    w_pixel, h_pixel = detection_info.get('w'), detection_info.get('h')
    if w_pixel is None or h_pixel is None or w_pixel <= 0 or h_pixel <= 0: return None
    inv_sqrt_area = 1 / math.sqrt(w_pixel * h_pixel)
    distance_cm = 18.5910 * math.log(inv_sqrt_area) + 107.7396
    return max(0, distance_cm)

# ==============================================================================
# 模块三：角度识别
# ==============================================================================
def find_two_largest_peaks_np(data):
    # (此函数无需修改，保持原样)
    peak_indices = np.where(np.r_[False, data[1:] > data[:-1]] & np.r_[data[:-1] > data[1:], False])[0]
    if len(peak_indices) < 2:
        max_idx = np.argmax(data)
        return [max_idx, max_idx]
    top_two_indices = peak_indices[np.argsort(data[peak_indices])[-2:]]
    return top_two_indices

def recognize_lock_angle(original_img, lock_bbox_info):
    try:
        x, y, w, h = lock_bbox_info['x'], lock_bbox_info['y'], lock_bbox_info['w'], lock_bbox_info['h']
        cropped_img = original_img[y:y+h, x:x+w]
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        _, seg = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg, connectivity=8)
        if num_labels <= 1: return None, None
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_component_label = np.argmax(areas) + 1
        binary_image = np.zeros_like(seg, dtype=np.uint8)
        binary_image[labels == largest_component_label] = 255
        h_bin, w_bin = binary_image.shape
        moments = cv2.moments(binary_image)
        if moments["m00"] == 0: return None, None
        center_x, center_y = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])
        center_point = (center_x, center_y)
        projections = np.array([np.sum(cv2.warpAffine(binary_image, cv2.getRotationMatrix2D(center_point, angle, 1.0), (w_bin, h_bin))[:, center_x]) for angle in range(180)])
        peak1_angle, peak2_angle = find_two_largest_peaks_np(projections)
        if abs(peak1_angle - peak2_angle) > 90:
            bisector_angle = ((peak1_angle + peak2_angle + 180) / 2) % 180
        else:
            bisector_angle = (peak1_angle + peak2_angle) / 2
        final_axis_angle = 90 - bisector_angle
        keyhole_orientation = 90 - final_axis_angle
        if keyhole_orientation > 90: keyhole_orientation -= 180
        final_angle = -keyhole_orientation
        
        # 【修正点】: 使用 bool() 将 numpy.bool_ 转换为原生 bool
        is_original = bool(abs(final_angle) < 5.0)

        return round(final_angle, 2), is_original
    except Exception:
        return None, None

# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    model_path = "src/best.onnx"
    image_folder_path = "DataAll" 
    class_names = ['lock', 'key']
    all_results = {}
    print("正在加载 YOLOv5 模型，请稍候...")
    try:
        net = cv2.dnn.readNetFromONNX(model_path)
        print("模型加载成功。")
    except Exception as e:
        print(f"致命错误：无法加载模型 '{model_path}'. 错误信息: {e}")
        exit()
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(valid_extensions)]
    total_images = len(image_files)
    print(f"在文件夹 '{image_folder_path}' 中找到 {total_images} 张图片。")

    for i, filename in enumerate(image_files):
        img_path = os.path.join(image_folder_path, filename)
        print(f"\n===== 处理图片: {filename} ({i+1}/{total_images}) =====")
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图片 {filename}，跳过。")
            all_results[filename] = {}
            continue
        detection_info = detect_objects(img, net, class_names)
        if not detection_info or detection_info.get("w", 0) <= 0:
            print("未检测到锁孔，跳过。")
            all_results[filename] = {}
            continue
        distance = calculate_distance_geometric(detection_info)
        lock_angle, is_lock_original = recognize_lock_angle(img, detection_info)
        key_angle = lock_angle if lock_angle is not None and detection_info.get("is_key_in", False) else 0.0
        
        # 【修正点】: 确保所有存入字典的值都是 JSON 可序列化的原生类型
        final_output = {
            "x": int(detection_info.get("x", 0)),
            "y": int(detection_info.get("y", 0)),
            "w": int(detection_info.get("w", 0)),
            "h": int(detection_info.get("h", 0)),
            "distance": round(distance, 2) if distance is not None else 0.0,
            "is_lock_original": bool(is_lock_original) if is_lock_original is not None else False,
            "lock_angle": lock_angle if lock_angle is not None else 0.0,
            "is_key_in": bool(detection_info.get("is_key_in", False)),
            "key_angle": key_angle
        }
        
        all_results[filename] = final_output
        
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_image = total_time / total_images if total_images > 0 else 0
    print("\n\n===== 最终 JSON 输出 =====")
    print(json.dumps(all_results, indent=4))
    print("\n\n===== 性能报告 =====")
    print(f"总共处理图片数量: {total_images}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每张图片耗时: {avg_time_per_image:.3f} 秒")