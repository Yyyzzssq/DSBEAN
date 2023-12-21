

import cv2

# 读取原图像
img = cv2.imread("test_image/plant_ori/cat_1.JPG")

# 读取坐标信息
with open("test_image/label_txt/max_data/cat_1.txt", "r") as f:
    x1, y1, x2, y2 = map(int, map(float, f.readline().split(',')))


# 在原图像上切割指定坐标范围的图像
cropped_img = img[y1:y2, x1:x2]

# 保存切割后的图像到指定位置
cv2.imwrite("test_image/cut_image/cut_1.jpg", cropped_img)



# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
#
# img = cv.imread('cropped_image.jpg', 0)
# sobelx8u = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=5)
#
# sobelx64f = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
# abs_sobel64f = np.absolute(sobelx64f)
# sobel_8u = np.uint8(abs_sobel64f)
# plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(1, 3, 2), plt.imshow(sobelx8u, cmap='gray')
# plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
# plt.subplot(1, 3, 3), plt.imshow(sobel_8u, cmap='gray')
# plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
# plt.show()
