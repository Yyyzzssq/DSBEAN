import cv2
import numpy as np

# 加载图像
img = cv2.imread(r'F:\PyCharm_Workspace\opencv\demo\compare_backbone\D_cut_image\demo\155.jpg')

# 将图像转换为HSV颜色空间
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义颜色的范围
lower_red = np.array([0, 43, 0])
upper_red = np.array([50, 255, 255])

# 创建掩码
mask = cv2.inRange(hsv_img, lower_red, upper_red)

# 将掩码应用于原始图像
color_img = cv2.bitwise_and(img, img, mask=mask)

# 调整显示大小
scale_percent = 50 # 缩放比例
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
color_img_resized = cv2.resize(color_img, dim, interpolation = cv2.INTER_AREA)

# 显示结果
# cv2.imshow('Original Image', img)
cv2.imshow('Color Image', color_img_resized)
cv2.imwrite(r"F:\PyCharm_Workspace\opencv\demo\compare_backbone\E_set_color\demo\155.jpg", color_img_resized)
cv2.waitKey()
cv2.destroyAllWindows()

# img0 = cv2.imread(r"F:\PyCharm_Workspace\opencv\demo\image\calculate_area\method_3\image\setColor.jpg")
# img1 = cv2.resize(img0, dsize=None, fx=0.5, fy=0.5)
# img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# h, w = img1.shape[:2]
# print(h, w)
#
#
# ret1, img4 = cv2.threshold(img2, 30, 255, cv2.THRESH_BINARY_INV)
# print(ret1)
#
# img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
# cv2.imshow("res", img4)
# cv2.imwrite(r"F:\PyCharm_Workspace\opencv\demo\image\calculate_area\method_3\image\result.jpg", img4)

#
#
# # 将二值图像转换为灰度图像
# gray_img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
#
# # 计算黑色像素占所有像素的比例
# total_pixels = h * w
# black_pixels = np.sum(gray_img4 == 0)
# black_ratio = black_pixels / total_pixels
# print(f"黑色像素占所有像素的比例为: {black_ratio}")


# 取色器
# http://www.jiniannet.com/Page/allcolor
# 将获取的四个点的rgb值填入下方的rgb，即可得到四个点的hsv上下限
# import cv2
# import numpy as np
#
# rgb = '#B18754,#E1CBA4,#D8BC8D,#F9F8F6'
#
# rgb = rgb.split(',')
#
# # 转换为BGR格式，并将16进制转换为10进制
# bgr = [[int(r[5:7], 16), int(r[3:5], 16), int(r[1:3], 16)] for r in rgb]
#
# # 转换为HSV格式
# hsv = [list(cv2.cvtColor(np.uint8([[b]]), cv2.COLOR_BGR2HSV)[0][0]) for b in bgr]
#
# hsv = np.array(hsv)
# print('H:', min(hsv[:, 0]), max(hsv[:, 0]))
# print('S:', min(hsv[:, 1]), max(hsv[:, 1]))
# print('V:', min(hsv[:, 2]), max(hsv[:, 2]))
