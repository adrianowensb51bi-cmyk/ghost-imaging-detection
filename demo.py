"""Quick demo to test YOLOv10DETR training with minimal epochs."""
import cv2

from ultralytics import YOLOv10

if __name__ == '__main__':
    # 创建YOLOv10DETR模型
    model = YOLOv10('./runs/detect/train3/weights/best.pt')

    # model.train(
    #     data='./ultralytics/cfg/datasets/citypersons.yaml',
    #     epochs=60,  # 只训练1个epoch用于测试
    #     batch=4,  # 小批次
    #     workers=1,
    #     imgsz=1024,
    #     val=True,  # 启用验证以测试验证流程
    # )
    image = r'C:\Users\VINTA\Desktop\文件夹\毕业论文\Graduate\GI\Ghost_imaging\results\predict\1.2'
    results = model(image, save=True, conf=0.3, imgsz=1024)  # predict on an image
