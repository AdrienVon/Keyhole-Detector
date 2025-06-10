import os
import time
import json

# ==============================================================================
# 模块导入与全局实例创建
# ==============================================================================
# 此部分在模块被导入时执行一次，实现高效的单例模式。
# ------------------------------------------------------------------------------
try:
    # 从您的核心逻辑文件中导入 LockAnalyzer 类
    from main_refactored import LockAnalyzer
except ImportError as e:
    # 这是一个备用方案，以防评测系统的路径设置有些特殊
    print(f"FATAL: 无法导入 LockAnalyzer。请确保 main_refactored.py 存在。错误: {e}")
    LockAnalyzer = None

GLOBAL_ANALYZER = None
if LockAnalyzer:
    try:
        # 创建全局唯一的分析器实例，模型在此处加载一次。
        GLOBAL_ANALYZER = LockAnalyzer()
    except Exception as e:
        print(f"FATAL: 分析器初始化失败。错误: {e}")
# ------------------------------------------------------------------------------


# ==============================================================================
# 比赛要求的接口函数
# ==============================================================================
def process_img(img_path):
    """
    比赛要求的处理函数接口。
    此函数调用全局分析器实例来处理图片。
    
    参数:
       img_path: 要识别的图片的路径
    
    返回:
       返回一个字典，包含所有识别结果。
    """
    if GLOBAL_ANALYZER is None:
        # 如果初始化失败，打印错误并返回空结果
        print(f"Error: Analyzer was not initialized. Cannot process image {os.path.basename(img_path)}")
        return {}
        
    try:
        # 调用核心分析逻辑
        result = GLOBAL_ANALYZER.analyze_image(img_path)
        return result
    except Exception as e:
        # 捕获处理单张图片时可能发生的未知异常
        print(f"Error: An error occurred while processing {os.path.basename(img_path)}: {e}")
        return {}


# ==============================================================================
# 官方提供的本地测试代码
# (最终提交时，评测系统不会运行这部分，但保留它用于本地验证非常方便)
# ==============================================================================
if __name__ == '__main__':
    # 1. --- 检查基本环境 ---
    try:
        from config import IMAGE_FOLDER_PATH
    except ImportError:
        print("警告: 无法从 config.py 导入 IMAGE_FOLDER_PATH，使用默认值 './Data/'")
        IMAGE_FOLDER_PATH = './Data/'
    
    imgs_folder = IMAGE_FOLDER_PATH
    
    if not os.path.isdir(imgs_folder):
        print(f"测试错误：测试图片文件夹 '{imgs_folder}' 不存在！")
        exit()

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    img_paths = [f for f in os.listdir(imgs_folder) if f.lower().endswith(valid_extensions)]
    
    if not img_paths:
        print(f"警告：在文件夹 '{imgs_folder}' 中没有找到任何图片文件。")
        exit()
    
    print(f"在 '{imgs_folder}' 中找到 {len(img_paths)} 张图片，准备开始处理...")
    
    all_results = {}
    
    # 2. --- 循环处理并收集结果 ---
    for filename in img_paths:
        full_path = os.path.join(imgs_folder, filename)
        
        # 调用核心处理函数
        individual_result = process_img(full_path)
        
        # 无论结果如何，都将其存入最终的字典中
        all_results[filename] = individual_result

    # 3. --- 最终结果输出 ---
    print("\n\n" + "="*25 + " 最终 JSON 输出 " + "="*25)
    
    # 使用 json.dumps 进行格式化打印
    print(json.dumps(all_results, indent=4))

    # 4. --- 性能报告 (可选，本地测试时很有用) ---
    # 注意：这里的计时并不精确，因为它包含了打印等 I/O 操作
    # 但可以作为一个粗略的性能参考
    print("\n\n" + "="*25 + " 性能报告 (仅供参考) " + "="*25)
    # 重新计时以获得更纯粹的计算时间
    start_perf_time = time.time()
    for filename in img_paths:
        full_path = os.path.join(imgs_folder, filename)
        process_img(full_path)
    end_perf_time = time.time()

    total_time = end_perf_time - start_perf_time
    total_images = len(img_paths)
    avg_time_per_image = total_time / total_images if total_images > 0 else 0
    
    print(f"总共处理图片数量: {total_images}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每张图片耗时: {avg_time_per_image * 1000:.2f} 毫秒")