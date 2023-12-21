import cv2
import numpy as np

img0 = cv2.imread(r"F:\PyCharm_Workspace\opencv\demo\compare_backbone\E_set_color\demo\155.jpg")
img1 = cv2.resize(img0, dsize=None, fx=0.5, fy=0.5)
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
h, w = img1.shape[:2]
# print(h, w)


ret1, img4 = cv2.threshold(img2, 30, 255, cv2.THRESH_BINARY_INV)
print(ret1)

img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
# cv2.imshow("res", img4)
cv2.imwrite(r"F:\PyCharm_Workspace\opencv\demo\compare_backbone\F_last_image\demo\155.jpg", img4)
cv2.waitKey()
cv2.destroyAllWindows()


# 将二值图像转换为灰度图像
gray_img4 = cv2.cvtColor(img4, cv2.COLOR_RGB2GRAY)

# 计算黑色像素占所有像素的比例
total_pixels = gray_img4.size  # 使用size属性来获取像素总数
black_pixels = np.sum(gray_img4 == 0)
black_ratio = black_pixels / total_pixels
print(f"黑色像素占所有像素的比例为: {black_ratio:.2%}")
