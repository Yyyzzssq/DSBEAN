import numpy as np
import cv2
import os

# 图像所在的文件夹路径
folder_path = r"F:\PyCharm_Workspace\opencv\demo\image\demo_stalk\test\im"


# 获取文件夹中所有图像的文件名
image_names = os.listdir(folder_path)
ruler_ac_len = 16
ruler_ac_wid = 1.8
ruler=(0,127,255)
caulis=(0,255,0)

# 定义颜色列表
# colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
ruler_wid = 1816

# 循环处理每张图像
for i, image_name in enumerate(image_names):
    # 读取图像和分割结果
    img_path = os.path.join(folder_path, image_name)
    img = cv2.imread(img_path)
    seg_result_path = os.path.join("image/demo_stalk/test/mask", image_name[:-4] + ".png")
    seg_result = cv2.imread(seg_result_path, 0)

    # 将分割结果二值化
    ret, thresh = cv2.threshold(seg_result, 0, 255, cv2.THRESH_BINARY)

    # 进行轮廓检测
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    # 绘制旋转矩形框
    for j, contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算矩形框的旋转角度
        # 计算矩形框的旋转角度和边长
        width = int(rect[1][1])
        height = int(rect[1][0])
        angle = rect[2]
        if angle < -45:
            angle += 90
        if width < height:
            res = height
            height = width
            width = res
        # 绘制矩形框
        color = caulis if width / height > (ruler_ac_len / ruler_ac_wid) + 2 else ruler
        cv2.drawContours(img, [box], 0, color, 10)
        scale = ruler_ac_len / ruler_wid

        # with open("image/demo_stalk/test/txt/IMG_0853.txt", 'r') as f:
        #     line = f.readline()
        #     coords = [float(c) for c in line.split(',')]
        #     x1, y1, x2, y2, w, h = coords
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     font_scale = 5
        #     thickness = 5
        #     text = f'{w * scale:.1f}cm x {h * scale:.1f}cm'
        #     text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        #     text_x, text_y = int(x1), int(y1 - text_size[1])
        #     cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

        # 在矩形框上显示长和宽
        if color == ruler:
            ruler_wid = width
            scale = ruler_ac_len / ruler_wid
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 5
            thickness = 5
            text = f'{ruler_ac_len:.1f}cm x {ruler_ac_wid:.1f}cm'
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x, text_y = box[1] if angle > -45 else box[0]
            cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 5
            thickness = 5
            text = f'{width * scale:.1f}cm x {height * scale:.1f}cm'
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x, text_y = box[1] if angle > -45 else box[0]
            cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

    # 将图像缩小至适合屏幕大小
    scale_percent = 30 # 调整缩放比例
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    # 保存结果到指定位置
    result_path = os.path.join("image/demo_stalk/test/result", image_name[:-4] + "_result3.png")
    cv2.imwrite(result_path, resized_img)
    cv2.imshow("Image", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
