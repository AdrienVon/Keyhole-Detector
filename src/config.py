# ==============================================================================
# 应用配置中心
# ==============================================================================

# --- 文件与路径配置 ---
MODEL_PATH = "src/best.onnx"
IMAGE_FOLDER_PATH = "DataAll"

# --- YOLOv5 模型配置 ---
YOLO_INPUT_WIDTH = 640
YOLO_INPUT_HEIGHT = 640
YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
CLASS_NAMES = ['lock', 'key']

# --- 角度识别配置 ---
ANGLE_PREPROCESS_THRESHOLD = 20  # 用于二值化的灰度阈值
ANGLE_RESET_THRESHOLD = 5.0      # 判断锁孔是否归位的角度阈值 (度)

# --- 距离计算配置 (公式中的常量) ---
# 公式: distance = A * log(1/sqrt(w*h)) + B
DISTANCE_PARAM_A = 18.5910
DISTANCE_PARAM_B = 107.7396