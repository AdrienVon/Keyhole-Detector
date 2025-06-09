import cv2
import numpy as np
import json
import os
import math

def detect_objects(image_path, model_path, class_names, conf_threshold=0.25, iou_threshold=0.45):
    """
    使用 YOLOv5 ONNX 模型检测图片中的目标，并返回指定格式的检测结果。

    Args:
        image_path (str): 输入图片的完整路径。
        model_path (str): ONNX 模型的完整路径。
        class_names (list): 类别名称列表，顺序需与模型训练时一致。
        conf_threshold (float): 置信度阈值，低于此值的检测将被过滤。
        iou_threshold (float): 非极大值抑制 (NMS) 的 IoU 阈值。

    Returns:
        dict: 格式为 {"image_filename": {"x": ..., "y": ..., "w": ..., "h": ..., "is_key_in": False}}
              如果图片中没有检测到 "lock" 或 "key" 类别，则对应的文件名下可能为空字典。
    """
    
    net = cv2.dnn.readNetFromONNX(model_path)

    # 尝试使用 GPU 加速 (如果你的系统支持并安装了 onnxruntime-gpu)
    # try:
    #     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    # except Exception as e:
    #     print(f"Warning: Could not set CUDA backend/target. Using CPU. Error: {e}")
    #     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return {os.path.basename(image_path): {}} # 返回空字典表示没有检测结果

    h_orig, w_orig, _ = img.shape # 原始图像的尺寸
    input_height, input_width = 640, 640 # YOLOv5 模型输入尺寸

    # --- 预处理图像 (letterbox) ---
    scale = min(input_width / w_orig, input_height / h_orig)
    new_w = int(w_orig * scale)
    new_h = int(h_orig * scale)

    padded_img = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    padded_img[(input_height - new_h) // 2 : (input_height - new_h) // 2 + new_h, \
               (input_width - new_w) // 2 : (input_width - new_w) // 2 + new_w, :] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    blob = cv2.dnn.blobFromImage(padded_img, 1/255.0, (input_width, input_height), swapRB=True, crop=False)
    net.setInput(blob)

    # --- 进行推理 ---
    output_data = net.forward()

    # --- 后处理输出 ---
    boxes = []
    confidences = []
    class_ids = []

    for detection in output_data[0]:
        obj_conf = detection[4] # 目标置信度
        class_scores = detection[5:] # 类别分数
        
        # 计算最终置信度 (目标置信度 * 类别分数)
        final_scores = obj_conf * class_scores

        class_id = np.argmax(final_scores)
        confidence = final_scores[class_id]
        
        if confidence > conf_threshold:
            # 边界框坐标 (x_center, y_center, width, height) - 像素坐标在 640x640 letterbox 图像上
            x_center = detection[0]
            y_center = detection[1]
            width_pred = detection[2] 
            height_pred = detection[3] 

            # 计算左上角坐标
            x_top_left = int(x_center - width_pred / 2)
            y_top_left = int(y_center - height_pred / 2)

            boxes.append([x_top_left, y_top_left, int(width_pred), int(height_pred)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # 应用非极大值抑制 (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)

    # 存储最终结果
    results = {
        "x": 0, "y": 0, "w": 0, "h": 0, "is_key_in": False
    }
    
    detected_locks = [] # 存储检测到的锁孔信息
    detected_keys = []  # 存储检测到的钥匙信息

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i] # letterbox 图像上的像素坐标

            # --- 将坐标反向转换回原始图像尺寸 ---
            # 计算填充量
            pad_w = (input_width - new_w) / 2 # 在你的640x480到640x640的例子中，pad_w=0
            pad_h = (input_height - new_h) / 2 # 在你的640x480到640x640的例子中，pad_h=80

            # 消除填充的影响
            final_x_on_padded = x - pad_w
            final_y_on_padded = y - pad_h
            
            # 反向缩放（如果 scale != 1）
            final_x = int(final_x_on_padded / scale)
            final_y = int(final_y_on_padded / scale)
            final_w = int(w / scale)
            final_h = int(h / scale)

            # 确保坐标在原始图像范围内
            final_x = max(0, final_x)
            final_y = max(0, final_y)
            final_w = min(w_orig - final_x, final_w)
            final_h = min(h_orig - final_y, final_h)

            current_detection_info = {
                "x": final_x, 
                "y": final_y, 
                "w": final_w, 
                "h": final_h,
                "confidence": confidences[i],
                "class_id": class_ids[i]
            }

            if class_names[class_ids[i]] == 'lock':
                detected_locks.append(current_detection_info)
            elif class_names[class_ids[i]] == 'key':
                detected_keys.append(current_detection_info)

    # --- 判断 is_key_in 逻辑 ---
    is_key_in_flag = False
    if detected_keys:  # 只要检测到钥匙，is_key_in 就为 True
        is_key_in_flag = True
    
    # 假设我们只关心第一个检测到的锁孔的信息 (如果没有则取默认值)
    if detected_locks:
        # 你可以根据需求选择最高置信度的锁孔，或者最近的锁孔等
        main_lock_info = detected_locks[0] # 这里简化为取第一个检测到的锁孔
        results["x"] = main_lock_info["x"]
        results["y"] = main_lock_info["y"]
        results["w"] = main_lock_info["w"]
        results["h"] = main_lock_info["h"]
    
    results["is_key_in"] = is_key_in_flag

    return {os.path.basename(image_path): results}




def calculate_distance_geometric(detection_info, focal_length_pixels=546.22, real_diameter_mm=28, pixel_measure_type='short_side'):
    """
    使用几何法计算锁孔到相机的距离。
    
    现在使用对数公式: distance = 18.5910 * log(1/sqrt(w*h)) + 107.7396

    Args:
        detection_info (dict): 单个锁孔的检测信息，包含 'w' (宽度) 和 'h' (高度)。
                                例如: {"x": ..., "y": ..., "w": 100, "h": 90}
        focal_length_pixels (float): 未使用，保留参数以兼容旧代码
        real_diameter_mm (float): 未使用，保留参数以兼容旧代码
        pixel_measure_type (str): 未使用，保留参数以兼容旧代码

    Returns:
        float: 锁孔到相机的距离 (厘米)，如果无法计算则返回 None。
    """
    if not isinstance(detection_info, dict) or 'w' not in detection_info or 'h' not in detection_info:
        # print("Error: detection_info must be a dictionary containing 'w' and 'h'.")
        return None
        
    w_pixel = detection_info.get('w')
    h_pixel = detection_info.get('h')

    if w_pixel is None or h_pixel is None or w_pixel <= 0 or h_pixel <= 0:
        # 边界框信息无效
        return None

    # 计算 1/sqrt(w*h)
    inv_sqrt_area = 1 / math.sqrt(w_pixel * h_pixel)
    
    # 使用对数公式计算距离: 18.5910 * log(1/sqrt(w*h)) + 107.7396
    distance_cm = 18.5910 * math.log(inv_sqrt_area) + 107.7396
    
    # 确保距离为正值
    if distance_cm < 0:
        distance_cm = 0
    
    return distance_cm





def recognize_lock_angle(image_path, lock_bbox_info, visualize=False):
    return None











# --- 测试模块化的函数 ---
if __name__ == "__main__":
    # 配置你的路径
    model_path = "src/best.onnx"
    # 使用指定的测试图片
    image_paths = [
        "src/1748242778221.jpg"  # 使用要求的图片路径
    ]
    class_names = ['lock', 'key']  # 你的类别名称

    all_detection_results = {}

    for img_path in image_paths:
        print(f"正在处理图片: {img_path}")
        result = detect_objects(img_path, model_path, class_names)
        
        # 如果检测到了锁孔，尝试计算距离
        filename = os.path.basename(img_path)
        if filename in result and result[filename] and result[filename]["w"] > 0:
            distance = calculate_distance_geometric(result[filename])
            if distance:
                # 在JSON中添加distance字段(在h和is_key_in之间)
                result_dict = result[filename]
                # 创建新字典包含重新排序的字段
                new_dict = {
                    "x": result_dict["x"],
                    "y": result_dict["y"],
                    "w": result_dict["w"],
                    "h": result_dict["h"],
                    "distance": round(distance, 2),  # 保留两位小数
                    "is_key_in": result_dict["is_key_in"]
                }
                result[filename] = new_dict
            
        all_detection_results.update(result)

    # 打印最终的 JSON 格式结果
    print(json.dumps(all_detection_results, indent=4))


