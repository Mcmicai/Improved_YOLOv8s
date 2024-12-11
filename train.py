import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from ultralytics import YOLO

def freeze_model(trainer):
    """
    回调函数：用于冻结模型的指定层。
    """
    model = trainer.model
    print("Before Freeze:")
    for k, v in model.named_parameters():
        print(f"\t{k}\t{'Trainable' if v.requires_grad else 'Frozen'}")

    # 冻结前10层
    freeze_layers = 10
    layers_to_freeze = [f"model.{x}." for x in range(freeze_layers)]
    
    for k, v in model.named_parameters():
        # 默认设置为可训练
        v.requires_grad = True
        # 冻结指定层
        if any(layer in k for layer in layers_to_freeze):
            print(f"Freezing layer: {k}")
            v.requires_grad = False

    print("After Freeze:")
    for k, v in model.named_parameters():
        print(f"\t{k}\t{'Trainable' if v.requires_grad else 'Frozen'}")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()

    # 加载预训练的 YOLOv8 模型
    # 模型可以通过 YAML 文件配置，也可以直接加载权重文件
    model = YOLO("yolov8s-JD-C2f-MSAM3.yaml").load("yolov8s.pt")
    #model = YOLO("yolov5s.pt")

    # 添加冻结层的回调函数
    model.add_callback("on_pretrain_routine_start", freeze_model)

    # 开始训练
    results = model.train(
    data="datasets/data.yaml",
    epochs=200,
    batch=8,
)

