- - # Keyhole Inspector - 自动化锁孔分析系统

    Keyhole Inspector 是一个基于 Python 和 OpenCV 的高级计算机视觉项目，旨在实现对锁孔图像的自动化分析。它利用内置的 YOLOv5 深度学习模型进行目标检测，并结合传统图像处理算法，精确计算锁孔的距离和旋转角度。

    ## 主要功能

    - **目标检测**: 使用内置的 YOLOv5 ONNX 模型快速定位图像中的“锁孔 (lock)”和“钥匙 (key)”。
    - **距离估算**: 基于检测框的大小，通过拟合的对数公式估算相机到锁孔的距离。
    - **角度识别**: 精确计算锁孔相对于垂直方向的旋转角度，精度高，鲁棒性强。
    - **状态判断**: 自动判断锁孔是否插入钥匙 (`is_key_in`) 以及锁孔是否处于归位状态 (`is_lock_original`)。
    - **灵活的输入**: 支持通过命令行参数对单个图片或整个文件夹进行分析。

    ## 项目架构简介

    本项目采用模块化和面向对象的设计，以实现高度的灵活性和可维护性。

    ```
    KEYHOLE-DETECTOR/
    ├── src/
    │   ├── best.onnx             # 内置的 YOLOv5 目标检测模型
    │   ├── config.py             # 核心配置文件，管理算法参数
    │   ├── main_refactored.py    # 项目的核心逻辑库
    │   └── process.py            # 本地快速测试与比赛接口适配脚本
    │
    ├── README.md                 # 本说明文件
    └── requirements.txt          # 项目依赖文件
    ```

    - **`src/config.py`**: 全局配置中心。管理算法参数，如检测阈值和公式常量。**项目的核心行为通过修改此文件来调整。**
    - **`src/main_refactored.py`**: 项目的核心逻辑库，包含 `LockAnalyzer` 类。
    - **`process.py`**: **用于快速启动和测试的脚本**。它调用核心库的功能，并提供一个简单的命令行接口。

    ## 快速启动本地测试

    本指南将帮助您快速设置本地开发环境，并使用 `process.py` 脚本对图片进行分析。

    ### 1. 设置本地开发环境

    **a. 克隆项目**

    ```bash
    git clone https://github.com/AdrienVon/Keyhole-Detector.git
    cd Keyhole-Detector
    ```

    **b. 创建并激活 Python 虚拟环境 (推荐)**

    - **Windows**:
      ```bash
      python -m venv venv
      .\venv\Scripts\activate
      ```
    - **macOS / Linux**:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

    **c. 安装依赖**

    项目所需的所有依赖项都已在 `requirements.txt` 中列出。

    ```bash
    pip install -r requirements.txt
    ```

    ### 2. 使用 `process.py` 脚本进行测试

    `process.py` 脚本通过**命令行参数**来指定要分析的图片或文件夹，**用户无需修改任何代码**。

    #### 方式一：测试指定的单个图片

    这是最常见的用法。将待测图片的路径作为参数传入。

    ```bash
    # 示例：
    python src/process.py path/to/your/image.jpg
    ```
    程序将只分析这张图片，并立即在控制台打印出该图片的结果。

    #### 方式二：测试指定的文件夹

    您也可以提供一个文件夹的路径，脚本会自动处理该文件夹下的所有图片。

    ```bash
    # 示例：
    python src/process.py path/to/your/image_folder/
    ```
    程序将逐一处理文件夹中的所有图片，并在最后打印出完整的 JSON 汇总结果和性能报告。

    ### 3. 理解输出结果

    脚本运行后，会在控制台打印出分析结果，最终输出一个完整的 JSON 对象，格式如下：

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
            "is_lock_original": true,
            "lock_angle": 2.3,
            "is_key_in": false,
            "key_angle": 0.0
        }
    }
    ```

    - **`x, y, w, h`**: 锁孔在原图中的边界框信息。
    - **`distance`**: 估算的距离（厘米）。
    - **`is_lock_original`**: 锁孔是否归位（旋转角度绝对值 < 5°）。
    - **`lock_angle`**: 锁孔相对于垂直方向的旋转角度。
    - **`is_key_in`**: 是否检测到钥匙。
    - **`key_angle`**: 如果有钥匙，则其角度与锁孔角度相同；否则为0。

    通过以上步骤，您可以非常方便地在本地进行开发、测试和验证。