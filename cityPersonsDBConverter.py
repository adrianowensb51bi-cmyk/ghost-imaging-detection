import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# ================= 配置区域 =================
# CityPersons 原始数据集根目录 (修改为你自己的路径)
ORIGIN_ROOT = r'datasets\CityPersons'  # 建议在 Windows 下路径前加 r，或使用双反斜杠

# 输出的 YOLO 格式数据集目录
OUTPUT_ROOT = './datasets/CityPersons_YOLOv10'

# 类别映射
CLASS_MAP = {
    'pedestrian': 0,
    'rider': 1,
}


# ===========================================

def setup_directories():
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_ROOT, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, 'labels', split), exist_ok=True)


def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    xmin, ymin, w, h = box
    x_center = (xmin + w / 2.0) * dw
    y_center = (ymin + h / 2.0) * dh
    w = w * dw
    h = h * dh
    return (x_center, y_center, w, h)


def get_label_path(img_path_str):
    """
    鲁棒的路径转换函数：
    1. 将所有反斜杠 \ 统一转换为 /，方便替换
    2. 执行替换
    3. 最后转回系统默认路径分隔符
    """
    # 统一转为 forward slash 处理
    path_unified = img_path_str.replace('\\', '/')

    # 替换文件夹路径 images -> labels
    label_path = path_unified.replace('/images/', '/labels/')

    # 替换文件名后缀
    label_path = label_path.replace('_leftImg8bit.png', '_gtBboxCityPersons.json')

    # 转回当前系统的路径格式 (Windows下变回 \)
    return os.path.normpath(label_path)


def process_dataset():
    setup_directories()
    splits = ['train', 'val', 'test']

    for split in splits:
        print(f"正在处理: {split} 集...")
        img_dir = os.path.join(ORIGIN_ROOT, 'images', split)

        # 搜索图片
        image_files = sorted(list(Path(img_dir).rglob('*_leftImg8bit.png')))

        if not image_files:
            print(f"警告: 在 {img_dir} 下没有找到图片，请检查路径配置！")
            continue

        for img_path_obj in tqdm(image_files):
            img_path = str(img_path_obj)

            # === 修正点：使用新的路径获取函数 ===
            label_path = get_label_path(img_path)

            # 目标路径
            file_name = os.path.basename(img_path)
            dst_img_path = os.path.join(OUTPUT_ROOT, 'images', split, file_name)
            dst_label_path = os.path.join(OUTPUT_ROOT, 'labels', split, file_name.replace('.png', '.txt'))

            # 复制图片
            if not os.path.exists(dst_img_path):
                shutil.copy(img_path, dst_img_path)

            # 处理标签 (Test集除外)
            if split != 'test':
                if not os.path.exists(label_path):
                    # 如果仍然找不到，打印出脚本在找哪个路径，方便调试
                    print(f"缺失: {label_path}")
                    # 生成空文件
                    open(dst_label_path, 'w').close()
                    continue

                try:
                    with open(label_path, 'r') as f:
                        data = json.load(f)

                    img_h = data.get('imgHeight')
                    img_w = data.get('imgWidth')

                    if img_h is None or img_w is None:
                        # 如果json没宽高，尝试读取图片 (需要 pip install opencv-python)
                        import cv2
                        img = cv2.imread(img_path)
                        img_h, img_w = img.shape[:2]

                    yolo_lines = []
                    objects = data.get('objects', [])

                    for obj in objects:
                        label_name = obj.get('label')
                        if label_name in CLASS_MAP:
                            cls_id = CLASS_MAP[label_name]
                            bbox = obj.get('bbox')

                            # 简单的越界检查
                            if bbox[2] <= 0 or bbox[3] <= 0: continue

                            yolo_box = convert_box((img_w, img_h), bbox)
                            line = f"{cls_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}"
                            yolo_lines.append(line)

                    with open(dst_label_path, 'w') as f_out:
                        f_out.write('\n'.join(yolo_lines))

                except Exception as e:
                    print(f"错误 {label_path}: {e}")

    print(f"\n转换完成！数据已保存至: {os.path.abspath(OUTPUT_ROOT)}")
    create_yaml_file()


def create_yaml_file():
    yaml_content = f"""
path: {os.path.abspath(OUTPUT_ROOT)}
train: images/train
val: images/val
test: images/test
nc: {len(CLASS_MAP)}
names:
"""
    sorted_names = sorted(CLASS_MAP.items(), key=lambda x: x[1])
    for name, _ in sorted_names:
        yaml_content += f"  - {name}\n"

    with open(os.path.join(OUTPUT_ROOT, 'citypersons.yaml'), 'w') as f:
        f.write(yaml_content)


if __name__ == '__main__':
    process_dataset()
