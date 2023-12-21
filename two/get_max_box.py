import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from unet import Unet
import os

# def get_red_pixel_coordinates_with_width(r_image, caulis_len):
def get_red_pixel_coordinates_with_width(r_image):
    # Convert PIL image to numpy array
    img = np.array(r_image)

    # Find red pixels
    red_range = ([80, 0, 0], [140, 50, 40])
    mask = np.logical_and(img >= red_range[0], img <= red_range[1])
    red_pixels = np.where(np.all(mask, axis=-1))

    # Calculate minimum upper-left and maximum lower-right coordinates
    x_min = np.min(red_pixels[1])
    y_min = np.min(red_pixels[0])
    x_max = np.max(red_pixels[1])
    y_max = np.max(red_pixels[0])

    # Calculate width of detection frame
    length = abs(x_max - x_min)

    # Draw rectangle on NumPy array
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=20)

    # Add text to image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    thickness = 20
    text = f"Length: {length}"
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = x_min
    text_y = y_min - text_size[1]
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

    # Convert NumPy array back to PIL image
    r_image = Image.fromarray(img)
    caulis_len = length
    # Return modified PIL image
    return r_image, caulis_len


def get_pod_box(caulis_len, output_path):
    image = cv2.imread(output_path)
    img_name = os.path.basename(output_path)
    filename_without_ext = os.path.splitext(img_name)[0]

    txt_route = os.path.join("length_ratio", "txt", filename_without_ext + '.txt')

    # 加载预测结果
    with open(txt_route, 'r') as f:
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

    # 计算边框长度
    box_length = max_x - min_x

    # print(min_x, min_y, max_x, max_y)
    # 用一个边界框框住所有检测框
    cv2.rectangle(image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 0, 255), 20)

    # 在左上角显示边框长度
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    thickness = 20
    text = f"Length: {box_length:.2f}"
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = int(min_x)
    text_y = int(min_y - text_size[1])
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

    # 将图像缩小以适应屏幕大小
    scale_percent = 50  # 缩放比例为50%
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    print(box_length)
    print(caulis_len)
    # 显示图像
    # cv2.imshow("Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存显示的图像和大检测框的坐标信息到指定位置
    output_path_res = os.path.join("length_ratio", "result_img", img_name)
    cv2.imwrite(output_path_res, resized_image)
    print("长度比: {:.2f}%".format(box_length / caulis_len * 100))



    return 0



if __name__ == "__main__":

    unet = Unet()
    mode = "predict"
    count = False
    name_classes = ["background", "caulis"]

    test_interval = 100
    caulis_len = 0
    dir_origin_path = r"F:\data\simple_1/"
    dir_save_path = r"F:\data\out/"

    simplify = True

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            img_name = os.path.basename(img)
            print(img_name)
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = unet.detect_image(image, count=count, name_classes=name_classes)
                r_image, caulis_len = get_red_pixel_coordinates_with_width(r_image)
                # r_image.show()
                output_path = os.path.join("length_ratio", "caulis_img", img_name)
                r_image.save(output_path)
                get_pod_box(caulis_len, output_path)


    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = unet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                get_red_pixel_coordinates_with_width(r_image)
                r_image.save(os.path.join(dir_save_path, img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
