import base64
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
from labelme import utils


## 方法 1

import base64
import json
import os
import os.path as osp

import numpy as np
import PIL.Image
from labelme import utils


if __name__ == '__main__':
    # 1.labelme标注  保存的json标注文件目录
    json_dir = r"F:\PyCharm_Workspace\opencv\demo\various_cat\json"

    # 2.保存转换后的mask  以及image文件目录
    jpgs_path   = r"F:\PyCharm_Workspace\opencv\demo\various_cat\image"
    pngs_path   = r"F:\PyCharm_Workspace\opencv\demo\various_cat\mask"

    #  标签的类别classes ,注意需要根据自己数据集的类别修改
    classes     = ["_background_","caulis", "ruler"]
    # classes     = ["_background_","cat","dog"]

    # 如果输出的路径不存在，则自动创建这个路径
    if not osp.exists(jpgs_path):
        os.mkdir(jpgs_path)

    if not osp.exists(pngs_path):
        os.mkdir(pngs_path)


    for file_name in os.listdir(json_dir):
        file_path = os.path.join(json_dir,file_name)
        # 遍历json_file里面所有的文件，并判断这个文件是不是以.json结尾
        if  os.path.isfile(file_path) and file_name.endswith(".json"):
            data = json.load(open(file_path))

            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(json_dir, data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value

            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))

            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)


            PIL.Image.fromarray(img).save(osp.join(jpgs_path, file_name.split(".")[0]+'.jpg'))

            new = np.zeros([np.shape(img)[0],np.shape(img)[1]])
            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)
                new = new + index_all*(np.array(lbl) == index_json)

            utils.lblsave(osp.join(pngs_path, file_name.split(".")[0]+'.png'), new)
            print('Saved ' + file_name.split(".")[0] + '.jpg and ' + file_name.split(".")[0] + '.png')
