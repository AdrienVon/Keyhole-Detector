import cv2
import numpy as np

# 替换为你的 ONNX 模型文件路径
model_path = "src/best.onnx"
# 替换为你要推理的图片文件路径
image_path = "src/1748242734557.jpg" # 你提供的示例图片
# 你的类别名称，需要和训练时 dataset.yaml 中的 names 顺序一致
class_names = ['lock', 'key'] # 确认你的类别顺序和名称无误

# 加载 ONNX 模型
net = cv2.dnn.readNetFromONNX(model_path)

# 如果有 GPU 并且你想使用 GPU 加速 (确保安装了 onnxruntime-gpu)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# 否则，默认会使用 CPU

# 读取图片
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not load image {image_path}")
    exit()

h_orig, w_orig, _ = img.shape # 原始图像的尺寸

# YOLOv5 输入尺寸
input_height, input_width = 640, 640

# --- 预处理图像 (letterbox) ---
# 计算 letterbox 缩放比例和填充
scale = min(input_width / w_orig, input_height / h_orig)
new_w = int(w_orig * scale)
new_h = int(h_orig * scale)

padded_img = np.full((input_height, input_width, 3), 114, dtype=np.uint8) # 填充灰度值 114
# 将原始图像缩放到 new_w x new_h 并居中放置在 padded_img 中
padded_img[(input_height - new_h) // 2 : (input_height - new_h) // 2 + new_h, \
           (input_width - new_w) // 2 : (input_width - new_w) // 2 + new_w, :] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# OpenCV 的 blobFromImage 会自动处理图片的缩放、归一化和通道顺序
blob = cv2.dnn.blobFromImage(padded_img, 1/255.0, (input_width, input_height), swapRB=True, crop=False)
net.setInput(blob)

# --- 进行推理 ---
output_data = net.forward()

# --- 后处理输出 ---
boxes = []
confidences = []
class_ids = []

# output_data[0] 包含了所有预测结果，每一行是一个预测
# 遍历每个预测结果
for detection in output_data[0]: 
    # 提取目标置信度 (objectness score) 和类别分数
    obj_conf = detection[4] # 第5个元素是目标置信度
    class_scores = detection[5:] # 从第6个元素开始是类别分数

    # 计算最终的置信度 (目标置信度 * 类别分数)
    final_scores = obj_conf * class_scores

    # 找出分数最高的类别
    class_id = np.argmax(final_scores)
    confidence = final_scores[class_id]
    
    # 过滤低置信度预测
    if confidence > 0.25: # 设置置信度阈值 (可调整)
        # 边界框坐标 (x_center, y_center, width, height) - 这些是像素坐标
        x_center = detection[0]
        y_center = detection[1]
        width_pred = detection[2] 
        height_pred = detection[3] 

        # 计算左上角坐标
        x_top_left = int(x_center - width_pred / 2)
        y_top_left = int(y_center - height_pred / 2)

        # 添加到列表
        boxes.append([x_top_left, y_top_left, int(width_pred), int(height_pred)])
        confidences.append(float(confidence))
        class_ids.append(class_id)

# --- 应用非极大值抑制 (NMS) ---
iou_threshold = 0.45 # 与 YOLOv5 默认值一致 (可调整)
nms_score_threshold = 0.25 # NMS 前的过滤阈值，可以与上面的 confidence 阈值保持一致

# cv2.dnn.NMSBoxes 接受 (boxes, confidences, score_threshold, nms_threshold)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, nms_score_threshold, iou_threshold)

# 如果有检测结果
if len(indexes) > 0:
    for i in indexes.flatten(): # NMSBoxes返回的索引是二维数组，flatten()将其变为一维
        x, y, w, h = boxes[i] # 这里的x,y,w,h是相对于640x640 letterbox图像的像素坐标

        # --- 将坐标反向转换回原始图像尺寸 ---
        # 1. 消除填充的影响
        # 对于 X 坐标 (w_orig = 640, new_w = 640), 填充宽度 pad_w = (640-640)/2 = 0
        # 对于 Y 坐标 (h_orig = 480, new_h = 480), 填充高度 pad_h = (640-480)/2 = 80
        # 原始图像在 letterbox 图像中的起始Y坐标是 pad_h = 80

        # x 坐标不需要偏移
        final_x = int(x) 
        # y 坐标需要向上偏移 pad_h
        final_y = int(y - (input_height - new_h) / 2) 
        final_w = int(w)
        final_h = int(h)
        
        # 由于 scale=1，这里不需要额外的缩放。
        # 如果原始图像尺寸不是640x480，并且scale不是1，则需要：
        # final_x = int((x - (input_width - new_w) / 2) / scale)
        # final_y = int((y - (input_height - new_h) / 2) / scale)
        # final_w = int(w / scale)
        # final_h = int(h / scale)


        # 确保坐标在原始图像范围内 (避免超出图片边界)
        final_x = max(0, final_x)
        final_y = max(0, final_y)
        final_w = min(w_orig - final_x, final_w) # 确保宽度不超过图片边界
        final_h = min(h_orig - final_y, final_h) # 确保高度不超过图片边界


        # 绘制检测框
        color = (0, 255, 0)
        cv2.rectangle(img, (final_x, final_y), (final_x + final_w, final_y + final_h), color, 2)

        # 绘制标签
        label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.putText(img, label, (final_x, final_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 显示结果
cv2.imshow("Detected Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()