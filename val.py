import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from ultralytics import YOLO


if __name__ == '__main__':
    # 加载模型
    model = YOLO("D:/Projects/Improved_YOLOv8s/runs/detect/train12/weights/best.pt")  # 替换为您的模型路径，如果使用自定义模型，加载训练完成的权重文件

    # 执行验证
    results = model.val(
        data="datasets/data.yaml",  # 数据配置文件路径
        split="test",               # 指定测试集
        batch=8,                    # 批量大小
        imgsz=640                   # 输入图像大小
    )

    # 打印验证结果
    print("Validation Results:")
    print(results)
