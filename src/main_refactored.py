import cv2
import numpy as np
import json
import os
import math
import time
from config import * # 从配置文件导入所有配置

# ==============================================================================
# 辅助函数 (Utility Functions)
# ==============================================================================

def find_two_largest_peaks_np(data: np.ndarray) -> list:
    """
    使用 NumPy 向量化操作在1D数据中寻找两个最大的局部峰值。

    Args:
        data (np.ndarray): 一维投影数据。

    Returns:
        list: 包含两个最大峰值索引的列表。如果峰值不足两个，则返回重复的全局最大值索引。
    """
    # 找到所有局部峰值的索引（一个点比其左右邻居都大）
    # np.r_ 用于在布尔数组两端添加 False，以优雅地处理边界情况。
    peak_indices = np.where(np.r_[False, data[1:] > data[:-1]] & np.r_[data[:-1] > data[1:], False])[0]
    
    if len(peak_indices) < 2:
        # 鲁棒性处理：如果峰值不足2个，则返回全局最大值的索引两次，以简化后续逻辑。
        max_idx = np.argmax(data)
        return [max_idx, max_idx]
    
    # 根据峰值的高度（即投影值）对所有找到的峰值进行排序，并返回最高的两个峰值的索引。
    top_two_indices = peak_indices[np.argsort(data[peak_indices])[-2:]]
    return top_two_indices.tolist() # 转换为原生list

# ==============================================================================
# 主分析器类 (Core Analyzer Class)
# ==============================================================================

class LockAnalyzer:
    """
    封装了所有锁孔分析功能的的高级分析器。

    该类采用单例模式的思想（在主程序中只实例化一次），在初始化时加载
    重量级资源（如ONNX模型），以优化批量处理性能。
    """
    def __init__(self):
        """
        初始化分析器实例。此方法在创建对象时被调用一次。
        主要负责加载深度学习模型。
        """
        print("正在初始化 LockAnalyzer...")
        self.class_names = CLASS_NAMES
        try:
            self.net = cv2.dnn.readNetFromONNX(MODEL_PATH)
            print("YOLOv5 模型加载成功。")
        except Exception as e:
            # 如果模型加载失败，这是一个致命错误，应立即抛出异常。
            raise IOError(f"致命错误：无法加载模型 '{MODEL_PATH}'. 错误: {e}")

    def analyze_image(self, img_path: str) -> dict:
        """
        对单张图片执行完整的分析流程。这是该类的主要公共接口。

        Args:
            img_path (str): 待分析图片的完整路径。

        Returns:
            dict: 包含所有分析结果的字典。如果处理失败或未检测到目标，则返回空字典。
        """
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图片 {img_path}，跳过。")
            return {}

        # 步骤 1: 目标检测
        detection_info = self._detect_objects(img)
        if not detection_info or detection_info.get("w", 0) <= 0:
            print("未检测到锁孔，处理中止。")
            return {}
            
        # 步骤 2: 计算距离
        distance = self._calculate_distance(detection_info)
        
        # 步骤 3: 识别角度
        lock_angle, is_lock_original = self._recognize_angle(img, detection_info)
        
        # 步骤 4: 根据逻辑计算钥匙角度
        key_angle = lock_angle if lock_angle is not None and detection_info.get("is_key_in", False) else 0.0

        # 步骤 5: 封装并返回最终结果
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
        return final_output

    def _detect_objects(self, img: np.ndarray) -> dict:
        """私有方法：使用YOLOv5模型执行目标检测。"""
        # ... (内部实现细节的注释可以相对简洁，因为函数名已经很清晰)
        # 预处理：letterbox缩放和填充
        h_orig, w_orig, _ = img.shape
        scale = min(YOLO_INPUT_WIDTH / w_orig, YOLO_INPUT_HEIGHT / h_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        padded_img = np.full((YOLO_INPUT_HEIGHT, YOLO_INPUT_WIDTH, 3), 114, dtype=np.uint8)
        padded_img[(YOLO_INPUT_HEIGHT - new_h) // 2 : (YOLO_INPUT_HEIGHT - new_h) // 2 + new_h, \
                   (YOLO_INPUT_WIDTH - new_w) // 2 : (YOLO_INPUT_WIDTH - new_w) // 2 + new_w, :] = cv2.resize(img, (new_w, new_h))
        
        # 推理
        blob = cv2.dnn.blobFromImage(padded_img, 1/255.0, (YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT), swapRB=True, crop=False)
        self.net.setInput(blob)
        output_data = self.net.forward()
        
        # 后处理：解析输出、NMS、坐标转换
        boxes, confidences, class_ids = [], [], []
        for detection in output_data[0]:
            final_scores = detection[4] * detection[5:]
            class_id = np.argmax(final_scores)
            confidence = final_scores[class_id]
            if confidence > YOLO_CONF_THRESHOLD:
                x_center, y_center, width_pred, height_pred = detection[:4]
                boxes.append([int(x_center - width_pred / 2), int(y_center - height_pred / 2), int(width_pred), int(height_pred)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD)
        
        detected_locks, detected_keys = [], []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                pad_w, pad_h = (YOLO_INPUT_WIDTH - new_w) / 2, (YOLO_INPUT_HEIGHT - new_h) / 2
                final_x = int((x - pad_w) / scale)
                final_y = int((y - pad_h) / scale)
                final_w = int(w / scale)
                final_h = int(h / scale)
                final_x, final_y = max(0, final_x), max(0, final_y)
                final_w, final_h = min(w_orig - final_x, final_w), min(h_orig - final_y, final_h)
                detection_info = {"x": final_x, "y": final_y, "w": final_w, "h": final_h}
                if self.class_names[class_ids[i]] == 'lock':
                    detected_locks.append(detection_info)
                elif self.class_names[class_ids[i]] == 'key':
                    detected_keys.append(detection_info)
        
        results = {"x": 0, "y": 0, "w": 0, "h": 0, "is_key_in": False}
        if detected_locks:
            results.update(detected_locks[0])
        if detected_keys:
            results["is_key_in"] = True
        return results

    def _calculate_distance(self, detection_info: dict) -> float:
        """私有方法：根据检测框尺寸估算距离。"""
        w_pixel, h_pixel = detection_info.get('w'), detection_info.get('h')
        if w_pixel is None or h_pixel is None or w_pixel <= 0 or h_pixel <= 0: return None
        
        # 使用配置中的参数进行计算
        inv_sqrt_area = 1 / math.sqrt(w_pixel * h_pixel)
        distance_cm = DISTANCE_PARAM_A * math.log(inv_sqrt_area) + DISTANCE_PARAM_B
        return max(0, distance_cm)

    # 在 LockAnalyzer 类中

    def _recognize_angle(self, original_img: np.ndarray, lock_bbox_info: dict) -> tuple:
        """私有方法：执行核心的角度识别算法。"""
        try:
            # 1. & 2. & 3. 裁剪、预处理、计算投影 (这部分保持不变)
            x, y, w, h = lock_bbox_info['x'], lock_bbox_info['y'], lock_bbox_info['w'], lock_bbox_info['h']
            cropped_img = original_img[y:y+h, x:x+w]
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            _, seg = cv2.threshold(gray, ANGLE_PREPROCESS_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
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

            # 4. 寻找双峰并计算角平分线 (这部分保持不变)
            peak1_angle, peak2_angle = find_two_largest_peaks_np(projections)
            if abs(peak1_angle - peak2_angle) > 90:
                bisector_angle = ((peak1_angle + peak2_angle + 180) / 2) % 180
            else:
                bisector_angle = (peak1_angle + peak2_angle) / 2

            # ==================================================================
            # 步骤 5: 【核心修正】根据新的旋转规则解释角度
            # ==================================================================
            # 'bisector_angle' 是锁孔对称轴与图像水平方向的夹角 (范围 0-180)。
            # 由于锁孔只在 0-90° 顺时针旋转，这个角度直接对应了锁孔的旋转角度。
            # 但由于对称性，一个旋转了 theta 度的轴和一个旋转了 180-theta 度的轴
            # 在投影上是无法区分的。我们需要选择落在 0-90 区间内的那个解。
            
            # 例如，如果 bisector_angle 是 170°，它在物理上等同于 10° 的轴，
            # 因为旋转范围是 0-90°，所以真正的旋转角度是 10°。
            
            if bisector_angle > 90:
                final_angle = 180 - bisector_angle
            else:
                final_angle = bisector_angle

            # 6. 判断是否归位 (逻辑不变，但基于新的 final_angle)
            is_original = bool(abs(final_angle) < ANGLE_RESET_THRESHOLD)

            return round(final_angle, 2), is_original
        
        except Exception as e:
            # 在复杂计算中捕获任何可能的异常，保证程序的健壮性
            print(f"角度识别时发生错误: {e}")
            return None, None

# ==============================================================================
# 主程序入口 (Main Execution Block)
# ==============================================================================

if __name__ == "__main__":
    # 1. 初始化分析器 (加载模型等重量级资源)
    try:
        analyzer = LockAnalyzer()
    except IOError as e:
        print(e)
        exit()

    # 2. 查找要处理的图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    if not os.path.isdir(IMAGE_FOLDER_PATH):
        print(f"错误: 文件夹 '{IMAGE_FOLDER_PATH}' (在config.py中定义) 不存在。")
        exit()
    image_files = [f for f in os.listdir(IMAGE_FOLDER_PATH) if f.lower().endswith(valid_extensions)]
    total_images = len(image_files)
    print(f"在文件夹 '{IMAGE_FOLDER_PATH}' 中找到 {total_images} 张图片。")

    # 3. 循环处理每张图片
    all_results = {}
    for i, filename in enumerate(image_files):
        img_path = os.path.join(IMAGE_FOLDER_PATH, filename)
        print(f"\n===== 处理图片: {filename} ({i+1}/{total_images}) =====")
        
        # 调用分析器的核心公共方法
        result = analyzer.analyze_image(img_path)
        all_results[filename] = result
        
    # 4. 打印最终的JSON报告
    print("\n\n===== 最终 JSON 输出 =====")
    print(json.dumps(all_results, indent=4))