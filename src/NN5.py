# 相较于NN4加入了畸变矫正的环节

import cv2
import numpy as np
import os
import json

# --- 相机标定参数 (请根据你的实际标定结果填写) ---
# 相机内参矩阵 K
# 来自你提供的图片中的 "The camera matrix"
camera_matrix = np.array([
    [546.216, 0, 289.049],
    [0, 545.736, 259.409],
    [0, 0, 1]
], dtype=np.float32)

# 畸变系数 D
# 来自你提供的图片中的 "Distortion"
dist_coeffs = np.array([-0.369099, 0.301648, -0.001657, 0.003514, 0.0], dtype=np.float32)
# 注意：OpenCV的畸变模型通常还包含k3，如果你的模型有k3，需要在这里添加。
# 如果标定结果只有4个系数，那么k3默认为0即可。

# 假设锁孔的实际物理直径为 28mm。这个值需要根据你的实际测量而定。
REAL_LOCK_DIAMETER_MM = 28


def detect_objects_and_undistort_roi(image_path, model_path, class_names, conf_threshold=0.25, iou_threshold=0.45,
                                      camera_matrix=None, dist_coeffs=None):
    """
    使用 YOLOv5 ONNX 模型检测图片中的目标，保留原始检测结果，
    并对检测到的“lock”主体结构区域进行畸变矫正。

    Args:
        image_path (str): 输入图片的完整路径。
        model_path (str): ONNX 模型的完整路径。
        class_names (list): 类别名称列表，顺序需与模型训练时一致。
        conf_threshold (float): 置信度阈值，低于此值的检测将被过滤。
        iou_threshold (float): 非极大值抑制 (NMS) 的 IoU 阈值。
        camera_matrix (np.array): 相机内参矩阵，用于去畸变。
        dist_coeffs (np.array): 畸变系数，用于去畸变。

    Returns:
        tuple: (dict, dict, float) 格式为 (
            {"image_filename": {"x": ..., "y": ..., "w": ..., "h": ..., "is_key_in": False}}, # 原始检测结果
            {"image_filename": cropped_undistorted_roi_image}, # 裁剪并去畸变后的ROI图像 (针对锁孔)
            new_focal_length_pixels # 去畸变后的有效焦距，用于距离计算
        )
        如果图片中没有检测到 "lock" 或 "key" 类别，则对应的文件名下可能为空字典。
        cropped_undistorted_roi_image 在没有锁孔时可能为 None。
    """
    net = cv2.dnn.readNetFromONNX(model_path)

    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"Error: Could not load image {image_path}")
        return {os.path.basename(image_path): {}}, None, None

    h_orig, w_orig, _ = img_original.shape  # 原始图像的尺寸
    input_height, input_width = 640, 640  # YOLOv5 模型输入尺寸

    # --- 预处理图像 (letterbox) for YOLO ---
    scale = min(input_width / w_orig, input_height / h_orig)
    new_w_padded = int(w_orig * scale)
    new_h_padded = int(h_orig * scale)

    padded_img = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    padded_img[(input_height - new_h_padded) // 2 : (input_height - new_h_padded) // 2 + new_h_padded, \
               (input_width - new_w_padded) // 2 : (input_width - new_w_padded) // 2 + new_w_padded, :] = \
        cv2.resize(img_original, (new_w_padded, new_h_padded), interpolation=cv2.INTER_AREA)

    blob = cv2.dnn.blobFromImage(padded_img, 1/255.0, (input_width, input_height), swapRB=True, crop=False)
    net.setInput(blob)

    # --- 进行推理 ---
    output_data = net.forward()

    # --- 后处理输出 ---
    boxes = []
    confidences = []
    class_ids = []

    for detection in output_data[0]:
        obj_conf = detection[4]  # 目标置信度
        class_scores = detection[5:]  # 类别分数
        
        final_scores = obj_conf * class_scores

        class_id = np.argmax(final_scores)
        confidence = final_scores[class_id]
        
        if confidence > conf_threshold:
            x_center = detection[0]
            y_center = detection[1]
            width_pred = detection[2] 
            height_pred = detection[3] 

            x_top_left = int(x_center - width_pred / 2)
            y_top_left = int(y_center - height_pred / 2)

            boxes.append([x_top_left, y_top_left, int(width_pred), int(height_pred)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # 应用非极大值抑制 (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)

    # 存储最终结果 (原始图像上的检测结果)
    original_detection_results = {
        "x": 0, "y": 0, "w": 0, "h": 0, "is_key_in": False
    }
    
    detected_locks = []  # 存储检测到的锁孔信息
    detected_keys = []   # 存储检测到的钥匙信息

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]  # letterbox 图像上的像素坐标

            # --- 将坐标反向转换回原始图像尺寸 ---
            pad_w = (input_width - new_w_padded) / 2
            pad_h = (input_height - new_h_padded) / 2

            final_x_on_padded = x - pad_w
            final_y_on_padded = y - pad_h
            
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
    if detected_locks and detected_keys:
        for lock_info in detected_locks:
            lock_x, lock_y, lock_w, lock_h = lock_info['x'], lock_info['y'], lock_info['w'], lock_info['h']
            
            for key_info in detected_keys:
                key_center_x = key_info['x'] + key_info['w'] / 2
                key_center_y = key_info['y'] + key_info['h'] / 2

                if lock_x < key_center_x < (lock_x + lock_w) and \
                   lock_y < key_center_y < (lock_y + lock_h):
                    is_key_in_flag = True
                    break
            if is_key_in_flag:
                break
    
    # 假设我们只关心第一个检测到的锁孔的信息
    main_lock_bbox_orig = None
    if detected_locks:
        main_lock_info = detected_locks[0] 
        original_detection_results["x"] = main_lock_info["x"]
        original_detection_results["y"] = main_lock_info["y"]
        original_detection_results["w"] = main_lock_info["w"]
        original_detection_results["h"] = main_lock_info["h"]
        main_lock_bbox_orig = (main_lock_info["x"], main_lock_info["y"], 
                               main_lock_info["w"], main_lock_info["h"])
    
    original_detection_results["is_key_in"] = is_key_in_flag

    cropped_undistorted_roi_image = None
    new_focal_length_pixels = None # 初始化为None

    # --- 对裁剪区域进行畸变矫正 ---
    if main_lock_bbox_orig and camera_matrix is not None and dist_coeffs is not None:
        x_orig, y_orig, w_orig_bbox, h_orig_bbox = main_lock_bbox_orig

        # 确保裁剪区域在图像边界内
        x_orig = max(0, x_orig)
        y_orig = max(0, y_orig)
        w_orig_bbox = min(w_orig_bbox, img_original.shape[1] - x_orig)
        h_orig_bbox = min(h_orig_bbox, img_original.shape[0] - y_orig)

        # 1. 创建一个新的投影矩阵 (new_camera_matrix)，用于去畸变和校正
        # 这里我们希望矫正后的图像是矩形的，不包含黑边，且焦距最佳
        # roi参数可以为空，OpenCV会根据新的相机矩阵自动确定最佳区域
        new_camera_matrix_global, roi_global = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w_orig, h_orig), 0, (w_orig_bbox, h_orig_bbox) # 传入原始图尺寸和期望ROI尺寸
        )

        # 提取去畸变后的有效焦距
        new_focal_length_pixels = new_camera_matrix_global[0, 0] # fx 通常等于 fy

        # 2. 计算映射图 (map1, map2)
        # map1 和 map2 存储了从去畸变图像到原始畸变图像的像素映射关系
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix_global, 
            (w_orig, h_orig), cv2.CV_32FC1
        )

        # 3. 使用 remap 进行去畸变。注意：这里是对整个原始图像进行映射，
        #    然后我们再从中裁剪出感兴趣区域。
        undistorted_full_image = cv2.remap(img_original, map1, map2, cv2.INTER_LINEAR)

        # 4. 从去畸变后的完整图像中裁剪出对应的ROI
        # 裁剪的坐标也需要通过新的相机矩阵确定，通常和原始ROI对应的位置一样
        # 但在某些情况下，ROI在去畸变后会略有移动。
        # 最简单的方法是直接根据原始ROI的坐标和去畸变后的图像尺寸进行裁剪，
        # 这也是 most common.
        cropped_undistorted_roi_image = undistorted_full_image[y_orig : y_orig + h_orig_bbox,
                                                                 x_orig : x_orig + w_orig_bbox]
        
        # 也可以通过 ROI 区域来更精确裁剪，但 for simplicity, use original ROI.
        # x_roi, y_roi, w_roi, h_roi = roi_global
        # cropped_undistorted_roi_image = undistorted_full_image[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]

    return {os.path.basename(image_path): original_detection_results}, \
           {os.path.basename(image_path): cropped_undistorted_roi_image}, \
           new_focal_length_pixels


def calculate_distance_geometric(detection_info, focal_length_pixels, real_diameter_mm, pixel_measure_type='long_side'):
    """
    使用几何法计算锁孔到相机的距离。

    Args:
        detection_info (dict): 单个锁孔的检测信息，包含 'w' (宽度) 和 'h' (高度)。
                                (注意: 这里传入的 w/h 是去畸变后裁剪区域的像素尺寸，
                                 但我们实际上是假设这个w/h对应的是校正后的锁孔尺寸)
                                例如: {"x": ..., "y": ..., "w": 100, "h": 90}
        focal_length_pixels (float): 相机焦距，单位像素。此焦距应是去畸变后的图像对应的焦距。
        real_diameter_mm (float): 锁孔的实际物理直径，单位毫米。
        pixel_measure_type (str): 指定使用识别框的哪个边作为像素尺寸 'd'。
                                  可以是 'long_side' (长边) 或 'short_side' (短边)。
                                  默认为 'long_side'。

    Returns:
        float: 锁孔到相机的距离 (厘米)，如果无法计算则返回 None。
    """
    if not isinstance(detection_info, dict) or 'w' not in detection_info or 'h' not in detection_info:
        return None
        
    w_pixel = detection_info.get('w')
    h_pixel = detection_info.get('h')

    if w_pixel is None or h_pixel is None or w_pixel <= 0 or h_pixel <= 0:
        return None

    d_pixel = 0.0
    if pixel_measure_type == 'long_side':
        d_pixel = max(w_pixel, h_pixel)
    elif pixel_measure_type == 'short_side':
        d_pixel = min(w_pixel, h_pixel)
    else:
        print(f"Error: Unsupported pixel_measure_type '{pixel_measure_type}'. Use 'long_side' or 'short_side'.")
        return None

    if d_pixel <= 0:
        return None

    # 使用公式 Z = (f * D) / d 计算距离 (单位为毫米)
    distance_mm = (focal_length_pixels * real_diameter_mm) / d_pixel
    
    # 转换为厘米
    distance_cm = distance_mm / 10.0
    
    return distance_cm






# # --- 测试模块化的函数 ---
# if __name__ == "__main__":
#     # 配置你的路径
#     model_path = "src/best.onnx"
#     # 使用指定的测试图片
#     image_paths = [
#         "src/1748242778221.jpg"  # 使用要求的图片路径
#     ]
#     class_names = ['lock', 'key']  # 你的类别名称

#     all_detection_results = {}

#     for img_path in image_paths:
#         print(f"正在处理图片: {img_path}")
#         # 使用带畸变矫正的检测函数
#         original_results, roi_images, new_focal_length = detect_objects_and_undistort_roi(
#             img_path, 
#             model_path, 
#             class_names,
#             camera_matrix=camera_matrix,
#             dist_coeffs=dist_coeffs
#         )
        
#         # 如果检测到了锁孔，尝试计算距离
#         filename = os.path.basename(img_path)
#         if (filename in original_results and 
#             original_results[filename] and 
#             original_results[filename]["w"] > 0 and
#             roi_images[filename] is not None):
            
#             # 获取去畸变后ROI的尺寸用于距离计算
#             roi_h, roi_w = roi_images[filename].shape[:2]
#             roi_info = {"w": roi_w, "h": roi_h}
            
#             # 正确传递参数计算距离
#             distance = calculate_distance_geometric(
#                 roi_info, 
#                 focal_length_pixels=new_focal_length,
#                 real_diameter_mm=REAL_LOCK_DIAMETER_MM
#             )
            
#             if distance:
#                 # 在JSON中添加distance字段(在h和is_key_in之间)
#                 result_dict = original_results[filename]
#                 # 创建新字典包含重新排序的字段
#                 new_dict = {
#                     "x": result_dict["x"],
#                     "y": result_dict["y"],
#                     "w": result_dict["w"],
#                     "h": result_dict["h"],
#                     "distance": round(distance, 2),  # 保留两位小数
#                     "is_key_in": result_dict["is_key_in"]
#                 }
#                 original_results[filename] = new_dict
                
#             print(f"估算的锁孔距离: {distance:.2f} 厘米" if distance else "无法估算距离")
            
#         all_detection_results.update(original_results)

#     # 打印最终的 JSON 格式结果
#     print("\n检测结果:")
#     print(json.dumps(all_detection_results, indent=4))







def process_data_all_folder(data_folder, output_csv_path, model_path, class_names):
    """
    处理DataAll文件夹中所有图片，执行锁孔检测并将结果保存到CSV文件
    
    Args:
        data_folder (str): 包含图片的文件夹路径
        output_csv_path (str): 输出CSV文件的路径
        model_path (str): ONNX模型的路径
        class_names (list): 类别名称列表
    """
    import os
    import csv
    import cv2
    
    # 检查文件夹是否存在
    if not os.path.exists(data_folder):
        print(f"错误: 找不到文件夹 {data_folder}")
        return
        
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(data_folder):
        file_lower = file.lower()
        if any(file_lower.endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(data_folder, file))
    
    if not image_files:
        print(f"警告: 在 {data_folder} 中没有发现图片文件")
        return
    
    # 创建CSV文件并写入表头
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_name', 'x', 'y', 'w', 'h', 'distance', 'is_key_in'])
        
        # 处理每张图片
        for img_path in image_files:
            print(f"正在处理图片: {os.path.basename(img_path)}")
            
            # 使用畸变矫正的检测函数
            original_results, roi_images, new_focal_length = detect_objects_and_undistort_roi(
                img_path, 
                model_path, 
                class_names,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs
            )
            
            # 获取文件名
            filename = os.path.basename(img_path)
            
            # 检查是否检测到锁孔
            if (filename in original_results and 
                original_results[filename] and 
                original_results[filename]["w"] > 0 and
                roi_images.get(filename) is not None):
                
                # 获取去畸变后ROI的尺寸
                roi_h, roi_w = roi_images[filename].shape[:2]
                roi_info = {"w": roi_w, "h": roi_h}
                
                # 计算距离
                distance = calculate_distance_geometric(
                    roi_info, 
                    focal_length_pixels=new_focal_length,
                    real_diameter_mm=REAL_LOCK_DIAMETER_MM
                )
                
                # 获取检测结果
                result = original_results[filename]
                
                # 写入CSV
                distance_value = round(distance, 2) if distance else "未知"
                
                csv_writer.writerow([
                    filename,
                    result["x"],
                    result["y"], 
                    result["w"], 
                    result["h"],
                    distance_value,
                    result["is_key_in"]
                ])
                
                print(f"估算的锁孔距离: {distance_value} 厘米")
            else:
                # 图片中未检测到有效锁孔，写入空行
                csv_writer.writerow([filename, 0, 0, 0, 0, "未检测到", False])
                print(f"图片 {filename} 中未检测到有效锁孔")
    
    print(f"\n处理完成! 结果已保存到 {output_csv_path}")


# 在main函数中调用此函数
if __name__ == "__main__":
    # 配置路径
    model_path = "src/best.onnx"
    data_folder = "DataAll"  # 包含所有图片的文件夹
    output_csv_path = "detection_results.csv"  # 输出CSV文件路径
    class_names = ['lock', 'key']
    
    # 处理所有图片
    process_data_all_folder(data_folder, output_csv_path, model_path, class_names)