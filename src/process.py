import os
import time
import json
import sys # <--- 引入 sys 模块来读取命令行参数

# ==============================================================================
# 模块导入与全局实例创建 (这部分保持不变)
# ==============================================================================
try:
    from main_refactored import LockAnalyzer
except ImportError as e:
    print(f"FATAL: 无法导入 LockAnalyzer。请确保 main_refactored.py 存在。错误: {e}")
    LockAnalyzer = None

GLOBAL_ANALYZER = None
if LockAnalyzer:
    try:
        GLOBAL_ANALYZER = LockAnalyzer()
    except Exception as e:
        print(f"FATAL: 分析器初始化失败。错误: {e}")

# ==============================================================================
# 比赛要求的接口函数 (这部分保持不变)
# ==============================================================================
def process_img(img_path):
    if GLOBAL_ANALYZER is None:
        print(f"Error: Analyzer was not initialized. Cannot process image {os.path.basename(img_path)}")
        return {}
    try:
        result = GLOBAL_ANALYZER.analyze_image(img_path)
        return result
    except Exception as e:
        print(f"Error: An error occurred while processing {os.path.basename(img_path)}: {e}")
        return {}

# ==============================================================================
# 官方提供的本地测试代码 (已重构以支持命令行参数)
# ==============================================================================
if __name__ == '__main__':
    # --- 1. 确定要处理的目标路径 ---
    target_path = ""
    
    # 检查是否从命令行提供了路径参数
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        print(f"从命令行接收到目标路径: '{target_path}'")
    else:
        # 如果没有提供命令行参数，则从 config.py 读取默认文件夹路径
        try:
            from config import IMAGE_FOLDER_PATH
            target_path = IMAGE_FOLDER_PATH
            print(f"未提供命令行参数，使用 config.py 中的默认文件夹: '{target_path}'")
        except ImportError:
            print("警告: 未提供命令行参数，且无法从 config.py 导入配置。使用 './DataAll/' 作为后备。")
            target_path = './DataAll/'

    # --- 2. 检查路径有效性并获取图片列表 ---
    if not os.path.exists(target_path):
        print(f"错误：指定的路径 '{target_path}' 不存在！")
        exit()

    image_files_to_process = []
    if os.path.isdir(target_path):
        # 如果是文件夹，则收集所有图片文件
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files_to_process = [os.path.join(target_path, f) for f in os.listdir(target_path) if f.lower().endswith(valid_extensions)]
    elif os.path.isfile(target_path):
        # 如果是单个文件，则列表只包含这一个文件
        image_files_to_process = [target_path]
    
    if not image_files_to_process:
        print(f"在路径 '{target_path}' 中没有找到任何可处理的图片文件。")
        exit()

    print(f"准备处理 {len(image_files_to_process)} 张图片...")

    # --- 3. 循环处理并收集/打印结果 ---
    all_results = {}
    for img_path in image_files_to_process:
        filename = os.path.basename(img_path)
        print(f"\n===== 处理图片: {filename} =====")
        
        individual_result = process_img(img_path)
        
        # 将结果存入字典，并立即打印单个结果
        all_results[filename] = individual_result
        print("--- 单张图片分析结果 ---")
        print(json.dumps({filename: individual_result}, indent=4))

    # --- 4. 如果处理的是文件夹，则打印最终的汇总JSON和性能报告 ---
    if os.path.isdir(target_path) and len(image_files_to_process) > 1:
        print("\n\n" + "="*25 + " 最终 JSON 汇总输出 " + "="*25)
        print(json.dumps(all_results, indent=4))

        print("\n\n" + "="*25 + " 性能报告 (仅供参考) " + "="*25)
        start_perf_time = time.time()
        for img_path in image_files_to_process:
            process_img(img_path) # 重新运行以计时
        end_perf_time = time.time()

        total_time = end_perf_time - start_perf_time
        total_images = len(image_files_to_process)
        avg_time_per_image = total_time / total_images if total_images > 0 else 0
        
        print(f"总共处理图片数量: {total_images}")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均每张图片耗时: {avg_time_per_image * 1000:.2f} 毫秒")