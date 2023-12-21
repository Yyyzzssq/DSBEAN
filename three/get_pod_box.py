import numpy as np
from PIL import Image, ImageDraw

# 加载预测图片
image = Image.open(r'F:\PyCharm_Workspace\opencv\demo\compare_backbone\plant_ori\demo\155.JPG').convert('RGB')

# 加载预测结果
with open(r'F:\PyCharm_Workspace\opencv\demo\compare_backbone\A_label_txt\all_data\demo\155.txt', 'r') as f:
    lines = f.readlines()

results = []
for line in lines:
    result = line.strip().split()
    results.append(result)
# 统计所有小检测框的数量
num_boxes = len(results)
print(f"总共检测到 {num_boxes} 个小检测框")

# 获取检测框信息
boxes = np.array([result[1:5] for result in results], dtype=np.float32)

# 计算所有检测框的最小和最大 x 和 y 坐标
min_x = np.min(boxes[:, 0])
min_y = np.min(boxes[:, 1])
max_x = np.max(boxes[:, 2])
max_y = np.max(boxes[:, 3])

print(min_x, min_y, max_x, max_y)
# 创建一个新的图像，并用一个边界框框住所有检测框
draw = ImageDraw.Draw(image)
draw.rectangle([(min_x, min_y), (max_x, max_y)], outline=(255, 0, 0), width=20)
# width = max_x - min_x
# height = max_y - min_y
# area = width * height
# print(f"大矩形框的面积为：{area}")

# 保存显示的图像和大检测框的坐标信息到指定位置
image.save(r'F:\PyCharm_Workspace\opencv\demo\compare_backbone\C_pod_box\demo\155.JPG')
with open(r'F:\PyCharm_Workspace\opencv\demo\compare_backbone\A_label_txt\max_data\demo\155.txt', 'w') as f:
    f.write(f"{min_x},{min_y},{max_x},{max_y}")
# # 显示图像
# # image.show()
# import cv2
# import numpy as np
# from PIL import Image
#
# # 加载预测图片
# image = cv2.imread(r'F:\PyCharm_Workspace\opencv\demo\compare_backbone\plant_ori\img3.JPG')
#
# # 加载预测结果
# with open(r'F:\PyCharm_Workspace\opencv\demo\compare_backbone\A_label_txt\all_data\cat_10.txt', 'r') as f:
#     lines = f.readlines()
#
# results = []
# for line in lines:
#     result = line.strip().split()
#     results.append(result)
#
# # 统计所有小检测框的数量
# num_boxes = len(results)
# print(f"总共检测到 {num_boxes} 个小检测框")
#
# # 获取检测框信息
# boxes = np.array([result[1:5] for result in results], dtype=np.float32)
#
# # 计算所有检测框的最小和最大 x 和 y 坐标
# min_x = np.min(boxes[:, 0])
# min_y = np.min(boxes[:, 1])
# max_x = np.max(boxes[:, 2])
# max_y = np.max(boxes[:, 3])
#
# # 计算边框长度
# box_length = max(max_x - min_x, max_y - min_y)
#
# print(min_x, min_y, max_x, max_y)
# # 用一个边界框框住所有检测框
# cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 0, 255), 5)
#
# # 在左上角显示边框长度
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 5
# thickness = 5
# text = f"Length: {box_length:.2f}"
# text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
# text_x = int(min_x)
# text_y = int(min_y - text_size[1])
# cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
#
# # 将图像缩小以适应屏幕大小
# scale_percent = 50  # 缩放比例为50%
# width = int(image.shape[1] * scale_percent / 100)
# height = int(image.shape[0] * scale_percent / 100)
# dim = (width, height)
# resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#
# # 显示图像
# cv2.imshow("Image", resized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 保存显示的图像和大检测框的坐标信息到指定位置
# cv2.imwrite(r'F:\PyCharm_Workspace\opencv\demo\compare_backbone\C_pod_box\cat_10.JPG', resized_image)
# # with open(r'F:\PyCharm_Workspace\opencv\demo\compare_backbone\A_label_txt\max_data\cat_10.txt', 'w') as f:
# #     f.write(f"{min_x},{min_y},{max_x},{max_y}")
# #
# # # 显示图像
# # cv2.imshow('image', image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
