# Keyhole Inspector - 自动化锁孔分析系统

Keyhole Inspector 是一个基于 Python 和 OpenCV 的高级计算机视觉项目，旨在实现对锁孔图像的自动化分析。它利用 YOLOv5 深度学习模型进行目标检测，并结合传统图像处理算法，精确计算锁孔的距离和旋转角度。

## 主要功能

- **目标检测**: 使用 YOLOv5 ONNX 模型快速定位图像中的“锁孔 (lock)”和“钥匙 (key)”。
- **距离估算**: 基于检测框的大小，通过拟合的对数公式估算相机到锁孔的距离。
- **角度识别**: 精确计算锁孔相对于垂直方向的旋转角度，精度高，鲁棒性强。
- **状态判断**: 自动判断锁孔是否插入钥匙 (`is_key_in`) 以及锁孔是否处于归位状态 (`is_lock_original`)。
- **批量处理**: 支持对整个文件夹的图片进行批量分析，并生成统一的 JSON 格式报告。

## 项目架构

本项目采用模块化和面向对象的设计，具有良好的可扩展性和可维护性。

- **`src/config.py`**: 全局配置中心，所有可调参数（如路径、阈值、模型参数）均在此定义。
- **`src/main_refactored.py`**: 核心逻辑实现。
  - **`LockAnalyzer` 类**: 封装了所有分析功能。它在初始化时加载一次模型，避免了重复的性能开销。
    - `analyze_image()`: 对外暴露的统一分析接口。
    - `_detect_objects()`: 内部实现目标检测。
    - `_calculate_distance()`: 内部实现距离计算。
    - `_recognize_angle()`: 内部实现角度识别。
- **`Data/`**: 存放待分析的图片。

## 环境要求



## 快速启动流程

1.  **克隆项目**
    ```bash
    git clone https://github.com/your-username/Keyhole-Detector.git
    cd Keyhole-Detector
    ```

2.  **准备数据**
    
    - 将您的 `.onnx` 模型文件放置在 `src/` 目录下，并确保其名称与 `src/config.py` 中的 `MODEL_PATH` 配置一致（默认为 `best.onnx`）。
    - 将所有待分析的图片放入 `Data/`文件夹中。
    
3.  **检查配置**
    
    - 打开 `src/config.py` 文件。
    - 检查 `MODEL_PATH` 和 `IMAGE_FOLDER_PATH` 是否正确。
    - 根据您的模型，确认 `CLASS_NAMES` 列表是否正确无误。
    - 按需调整其他阈值参数。
    
4.  **运行分析**
    
    - 在项目根目录下，执行主程序：
    ```bash
    python src/main_refactored.py
    ```
    
5.  **查看结果**
    - 程序将开始逐一处理 `Data/` 文件夹中的图片，并在控制台打印处理进度。
    - 处理完成后，完整的分析结果将以 **JSON 格式**打印在控制台的末尾。
    - 同时，还会生成一份简短的性能报告，显示总耗时和平均每张图片的处理时间。

## 输出格式说明

最终的 JSON 输出格式如下：

```json
{
    "image1.jpg": {
        "x": 120,
        "y": 80,
        "w": 150,
        "h": 145,
        "distance": 30.52,
        "is_lock_original": false,
        "lock_angle": -15.78,
        "is_key_in": true,
        "key_angle": -15.78
    },
    "image2.png": {
        "x": 200,
        "y": 150,
        "w": 160,
        "h": 162,
        "distance": 28.1,
        "is_lock_original": true,
        "lock_angle": 2.3,
        "is_key_in": false,
        "key_angle": 0.0
    },
    "no_detection.jpg": {}
}
```

- `x, y, w, h`: 锁孔在原图中的边界框坐标和尺寸。
- `distance`: 估算的距离（单位：厘米）。
- `is_lock_original`: 锁孔是否归位（旋转角度绝对值 < 5°）。
- `lock_angle`: 锁孔相对于垂直方向的旋转角度（°）。
- `is_key_in`: 是否检测到钥匙。
- `key_angle`: 如果有钥匙，则其角度与锁孔角度相同；否则为0。