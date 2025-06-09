import cv2
import numpy as np

# 1. 读取您上传的图片
# 确保 'image_6dee54.png' 文件与此脚本在同一目录下
try:
    img = cv2.imread('src/angle/cutResult/keyOut/1748243267394_crop.jpg')
    if img is None:
        raise FileNotFoundError("无法找到或打开图片文件 'image_6dee54.png'")
except Exception as e:
    print(e)
    exit()

# 2. 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. 应用您指定的阈值分割（参数保持不变）
# cv2.THRESH_BINARY_INV 表示反向二值阈值，目标为白色(255)，背景为黑色(0)
_, seg = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)

# --- 以下是核心处理部分 ---

# 4. 使用连通组件分析
# cv2.connectedComponentsWithStats 会返回组件数量、一个标记了每个组件的图像、每个组件的统计信息和中心点
# connectivity=8 表示使用8连通域
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg, connectivity=8, ltype=cv2.CV_32S)

# 检查是否存在前景对象 (num_labels > 1)
if num_labels > 1:
    # 5. 找到最大连通组件（排除背景）
    # stats 的第一行 (索引0) 是背景的信息，所以我们从索引1开始查找
    areas = stats[1:, cv2.CC_STAT_AREA]  # 提取所有前景组件的面积
    
    # 找到面积最大的组件的索引。注意：这个索引是相对于 areas 数组的，
    # 所以需要加1来得到它在原始 labels 和 stats 中的真实标签值
    largest_component_label = np.argmax(areas) + 1

    # 6. 创建一个只包含最大组件的空白图像
    # 创建一个与原始二值图大小相同的全黑图像
    output_img = np.zeros_like(seg)
    # 将最大组件对应的像素位置设置为白色 (255)
    output_img[labels == largest_component_label] = 255
else:
    # 如果没有前景对象，输出就是全黑图像
    output_img = np.zeros_like(seg)


# --- 显示结果 ---
cv2.imshow('Original Segmented (seg)', seg)
cv2.imshow('Largest Component Only (output_img)', output_img)

# 等待按键后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

# 可选：保存处理后的结果
# cv2.imwrite('result.png', output_img)