import cv2
import numpy as np
import json
import os

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

    # --- 判断 is_key_in 逻辑 (TODO: 需要根据你的实际场景细化) ---
    is_key_in_flag = False
    if detected_locks and detected_keys:
        # 简化判断：如果检测到一个锁孔和一个钥匙，并且钥匙的中心点在锁孔内部
        # 遍历所有锁孔和钥匙，寻找最佳匹配
        for lock_info in detected_locks:
            lock_x, lock_y, lock_w, lock_h = lock_info['x'], lock_info['y'], lock_info['w'], lock_info['h']
            
            for key_info in detected_keys:
                key_center_x = key_info['x'] + key_info['w'] / 2
                key_center_y = key_info['y'] + key_info['h'] / 2

                # 检查钥匙中心点是否在锁孔内
                if lock_x < key_center_x < (lock_x + lock_w) and \
                   lock_y < key_center_y < (lock_y + lock_h):
                    is_key_in_flag = True
                    break # 找到一个符合条件的就退出钥匙循环
            if is_key_in_flag:
                break # 如果找到一个锁孔有钥匙，就退出锁孔循环
    
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




    """
    在锁孔的边界框内识别其内部结构的旋转角度。
    假设锁孔内部是“一个两边有缺口的圆环”，缺口部分是黑色。

    Args:
        image_path (str): 原始图片的路径。
        lock_bbox_info (dict): 从 detect_objects 返回的锁孔边界框信息
                                {"x": x, "y": y, "w": w, "h": h}。
        visualize (bool): 是否显示中间处理步骤的图像。

    Returns:
        float: 锁孔的旋转角度（例如，0-360度），或 None 如果无法识别。
    """
    if not lock_bbox_info or lock_bbox_info['w'] == 0:
        return None # 如果没有有效的锁孔框

    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Error: Could not load original image {image_path}")
        return None

    x, y, w, h = lock_bbox_info['x'], lock_bbox_info['y'], lock_bbox_info['w'], lock_bbox_info['h']

    # 1. 裁剪锁孔区域 (添加一些边距以防裁剪太紧)
    padding = 10 # 增加一些边距
    x_crop = max(0, x - padding)
    y_crop = max(0, y - padding)
    w_crop = min(img_orig.shape[1] - x_crop, w + 2 * padding)
    h_crop = min(img_orig.shape[0] - y_crop, h + 2 * padding)

    lock_roi = img_orig[y_crop : y_crop + h_crop, x_crop : x_crop + w_crop].copy()

    if visualize:
        cv2.imshow("Lock ROI", lock_roi)
        cv2.waitKey(1)

    if lock_roi.size == 0: # 检查裁剪区域是否为空
        return None

    # 2. 灰度化
    gray_roi = cv2.cvtColor(lock_roi, cv2.COLOR_BGR2GRAY)

    # 3. 阈值分割 (找到黑色缺口)
    # 根据你的图片情况调整阈值。通常缺口是黑色，背景是浅色。
    # cv2.THRESH_BINARY_INV 表示像素值低于阈值的设为白色（255），高于的设为黑色（0）
    # 或者直接 THRESH_BINARY 表示像素值低于阈值的设为黑色（0），高于的设为白色（255）
    # 你可能需要多次尝试来找到最佳阈值
    # 假设缺口是黑色，背景亮，我们想让缺口变成白色，便于寻找白色区域
    _, binary_roi = cv2.threshold(gray_roi, 60, 255, cv2.THRESH_BINARY_INV) # 假设缺口较暗，阈值60

    if visualize:
        cv2.imshow("Binary ROI", binary_roi)
        cv2.waitKey(1)

    # 4. 形态学操作（可选，用于去除噪声和连接缺口）
    kernel = np.ones((3, 3), np.uint8)
    binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, kernel, iterations=1) # 开运算，去除小点噪声
    binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel, iterations=1) # 闭运算，连接断裂部分

    if visualize:
        cv2.imshow("Morphed Binary ROI", binary_roi)
        cv2.waitKey(1)

    # 5. 寻找轮廓并分析 (找到缺口)
    # 假设缺口会形成明显的白色连通区域
    contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤小轮廓，保留大的潜在缺口区域
    min_缺口面积 = (w * h) * 0.005 # 最小缺口面积，根据实际情况调整
    缺口中心点s = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_缺口面积:
            # 计算轮廓的矩 (moments)，用于找到中心
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                缺口中心点s.append((cX, cY))
                
                if visualize:
                    # 在 ROI 上画出轮廓和中心点
                    cv2.drawContours(lock_roi, [cnt], -1, (0, 0, 255), 1) # 红色轮廓
                    cv2.circle(lock_roi, (cX, cY), 3, (0, 255, 255), -1) # 黄色中心点

    if visualize:
        cv2.imshow("Contours and Centers", lock_roi)
        cv2.waitKey(1)

    # 6. 根据缺口中心点计算角度
    if len(缺口中心点s) >= 2:
        # 假设找到的两个最大的区域就是两个缺口
        # 简化：取前两个找到的缺口中心点
        pt1 = 缺口中心点s[0]
        pt2 = 缺口中心点s[1]

        # 计算两点之间的连线向量
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]

        # 使用 atan2 计算角度（弧度），然后转换为度
        # atan2(dy, dx) 返回的角度范围是 -pi 到 pi
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        # 调整角度到 0-360 度范围，或根据你的需求调整基准
        # 例如，如果锁孔是水平时为0度，垂直时为90度
        # 你可能需要根据实际图像和缺口定义来调整
        angle_deg = (angle_deg + 360) % 360
        
        # 为了更直观，我们可以以垂直向上为0度（或者水平向右为0度，取决于你的定义）
        # 这里假设缺口连线与水平轴的夹角
        # 如果你希望竖直向上为0度，那么可以计算 angle_deg = (90 - angle_deg + 360) % 360
        
        return angle_deg
    else:
        print("Warning: Could not find two distinct gap regions in the lock ROI.")
        return None

def get_lock_inner_structure_info(image_path, lock_bbox_info, threshold_value=20, visualize=False):
    """
    在锁孔的边界框内，使用阈值分割和轮廓查找，识别锁孔内部的特定结构（缺口圆环），
    并返回该结构的最大轮廓及其几何中心。

    Args:
        image_path (str): 原始图片的完整路径。
        lock_bbox_info (dict): 从 detect_objects 返回的锁孔边界框信息
                                {"x": x, "y": y, "w": w, "h": h}。
        threshold_value (int): 用于二值化的阈值（0-255）。根据你的图片调整。
        visualize (bool): 是否显示中间处理步骤的图像。

    Returns:
        tuple: (max_contour, center_x_roi, center_y_roi, lock_roi_with_contour)
               max_contour (numpy.ndarray): 识别到的面积最大的轮廓（相对于 ROI 坐标）。
               center_x_roi (int): 轮廓在 ROI 中的中心点 X 坐标。
               center_y_roi (int): 轮廓在 ROI 中的中心点 Y 坐标。
               lock_roi_with_contour (numpy.ndarray): 绘制了最大轮廓的彩色 ROI 图像（用于调试可视化）。
               如果无法处理或未找到有效轮廓，则返回 (None, None, None, None)。
    """
    if not lock_bbox_info or lock_bbox_info['w'] == 0:
        print("Error: No valid lock bounding box info provided for inner structure detection.")
        return None, None, None, None

    img_orig_copy = cv2.imread(image_path) # 读取原始图像的副本，用于可视化
    if img_orig_copy is None:
        print(f"Error: Could not load original image {image_path}")
        return None, None, None, None

    x, y, w, h = lock_bbox_info['x'], lock_bbox_info['y'], lock_bbox_info['w'], lock_bbox_info['h']

    # 裁剪锁孔区域 (添加一些边距以防裁剪太紧)
    padding = 10 
    x_crop = max(0, x - padding)
    y_crop = max(0, y - padding)
    w_crop = min(img_orig_copy.shape[1] - x_crop, w + 2 * padding)
    h_crop = min(img_orig_copy.shape[0] - y_crop, h + 2 * padding)

    lock_roi = img_orig_copy[y_crop : y_crop + h_crop, x_crop : x_crop + w_crop].copy()
    lock_roi_display = lock_roi.copy() # 用于绘制轮廓的副本

    if lock_roi.size == 0:
        print("Error: Cropped lock ROI is empty for inner structure detection.")
        return None, None, None, None

    # 1. 转换为灰度图
    gray_roi = cv2.cvtColor(lock_roi, cv2.COLOR_BGR2GRAY)

    # 2. 阈值分割：保留接近黑色的部分（暗区域），去掉亮的区域
    # THRESH_BINARY_INV 表示：灰度 < 阈值 → 255（白），否则 0（黑）
    _, seg = cv2.threshold(gray_roi, threshold_value, 255, cv2.THRESH_BINARY_INV)

    if visualize:
        cv2.imshow("Lock ROI Original", lock_roi)
        cv2.imshow(f"Lock ROI Binary (Threshold={threshold_value})", seg)
        cv2.waitKey(1)

    # 3. 查找轮廓（只找外轮廓）
    # CHAIN_APPROX_NONE 存储所有轮廓点
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print("Warning: No contours found in the segmented lock ROI.")
        return None, None, None, None

    # 4. 找面积最大的一个轮廓（假设就是整个结构轮廓）
    max_contour = max(contours, key=cv2.contourArea)

    # 计算最大轮廓的几何中心
    M = cv2.moments(max_contour)
    center_x_roi, center_y_roi = None, None
    if M["m00"] != 0:
        center_x_roi = int(M["m10"] / M["m00"])
        center_y_roi = int(M["m01"] / M["m00"])
    
    if visualize:
        # 在 ROI 副本上绘制找到的最大轮廓
        cv2.drawContours(lock_roi_display, [max_contour], -1, (0, 255, 0), 2) # 绿色轮廓
        if center_x_roi is not None:
            cv2.circle(lock_roi_display, (center_x_roi, center_y_roi), 5, (0, 0, 255), -1) # 红色中心点
        cv2.imshow("Max Contour in ROI", lock_roi_display)
        cv2.waitKey(1)

    return max_contour, center_x_roi, center_y_roi, lock_roi_display


# --- 在你的 main 函数中调用 ---
if __name__ == "__main__":
    model_path = "src/best.onnx"
    image_paths = [
        "src/1748243361956.jpg", 
        # "src/lock.png", # 如果你的 lock.png 在 src 目录下
        # 其他测试图片
    ]
    class_names = ['lock', 'key'] 

    all_final_results = {}

    for img_path in image_paths:
        detection_output = detect_objects(img_path, model_path, class_names)
        
        image_filename = os.path.basename(img_path)
        final_result_for_image = detection_output.get(image_filename, {})

        if final_result_for_image and final_result_for_image['w'] > 0: # 确保检测到锁孔
            lock_bbox = {
                "x": final_result_for_image['x'],
                "y": final_result_for_image['y'],
                "w": final_result_for_image['w'],
                "h": final_result_for_image['h']
            }
            
            # 2. 在锁孔区域内进行阈值分割和轮廓查找
            # 调整 threshold_value 参数以找到最佳分割效果
            max_contour, roi_center_x, roi_center_y, lock_roi_with_contour = \
                get_lock_inner_structure_info(img_path, lock_bbox, threshold_value=20, visualize=True) 
            
            if max_contour is not None:
                print(f"Successfully found main inner structure for {image_filename}.")
                print(f"Center in ROI: ({roi_center_x}, {roi_center_y})")
                
                # TODO: 在这里添加后续的逻辑，根据 max_contour 计算锁孔的旋转角度。
                # max_contour 是一个包含所有轮廓点的 NumPy 数组。
                # 你可以尝试：
                # 1. 拟合最小外接矩形 (cv2.minAreaRect(max_contour))，它会返回旋转角度。
                #    这个角度是矩形的长边或短边与X轴的夹角，需要根据你的定义进行转换。
                # 2. 分析轮廓的凸包 (convex hull) 或凸缺陷 (convexity defects) 来识别缺口。
                # 3. 如果已知是圆环，可以尝试 Hough 圆检测 (cv2.HoughCircles) 找到圆心和半径。
                # 4. 如果有明确的两个缺口，找到它们在轮廓上的对应点，计算两点连线的角度。

            else:
                print(f"No main inner structure detected for lock in {image_filename}.")
                
            all_final_results[image_filename] = final_result_for_image 
        else:
            print(f"No lock detected in {image_filename}")

    print(json.dumps(all_final_results, indent=4))
    
    cv2.waitKey(0) 
    cv2.destroyAllWindows()





# # --- 测试模块化的函数 ---
# if __name__ == "__main__":
#     # 配置你的路径
#     model_path = "src/best.onnx"
#     # 假设你有一个测试图片列表
#     image_paths = [
#         "src/1748243297059.jpg", # 假设这张图片有锁孔和钥匙
#         # "src/another_image_with_lock.jpg", # 另一张图片
#         # "src/image_without_lock.jpg" # 没有锁孔的图片
#     ]
#     class_names = ['lock', 'key'] # 你的类别名称

#     all_detection_results = {}

#     for img_path in image_paths:
#         result = detect_objects(img_path, model_path, class_names)
#         all_detection_results.update(result)
        
#         # 可选：在函数内部或此处绘制结果用于调试
#         # 为了简洁，这里不再包含显示图片的逻辑，如果你需要可以再加入
#         # load image again for drawing
#         # img_display = cv2.imread(img_path)
#         # if os.path.basename(img_path) in all_detection_results:
#         #     det_info = all_detection_results[os.path.basename(img_path)]
#         #     if det_info['w'] > 0: # 确保有检测到框
#         #         x, y, w, h = det_info['x'], det_info['y'], det_info['w'], det_info['h']
#         #         cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         #         # Add label for lock
#         #         cv2.putText(img_display, "Lock", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#         #     if det_info['is_key_in']:
#         #         cv2.putText(img_display, "Key In", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         # cv2.imshow("Detected Image", img_display)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()


#     # 打印最终的 JSON 格式结果
#     print(json.dumps(all_detection_results, indent=4))
    
#     # 也可以保存到文件
#     # with open("detection_results.json", "w") as f:
#     #     json.dump(all_detection_results, f, indent=4)




