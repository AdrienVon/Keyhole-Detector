import cv2
import numpy as np
import math

# 读取图像
img = cv2.imread('lock.png')

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 阈值分割：保留接近黑色的部分（暗区域），去掉亮的区域
# THRESH_BINARY_INV 表示：灰度 < 阈值 → 255（白），否则 0（黑）
_, seg = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
# 展示图片
cv2.imshow("Binary", seg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 查找轮廓（只要外轮廓）
contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#展示轮廓
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
cv2.imshow("Contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 找面积最大的一个轮廓（假设就是整个结构轮廓）
contour = max(contours, key=cv2.contourArea)
#展示轮廓
cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
cv2.imshow("Max Contour", img)
cv2.waitKey(0)
cv2.destroyAllWindows()