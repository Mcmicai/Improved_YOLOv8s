import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from ultralytics import YOLO


def freeze_model(trainer):
    model = trainer.model
    print('Before Freeze')
    for k, v in model.named_parameters():
        print('\t', k, '\t', v.requires_grad)

    # 冻结前10层
    freeze = 10
    freeze = [f'model.{x}.' for x in range(freeze)]
    for k, v in model.named_parameters():
        v.requires_grad = True  # 训练所有层
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False

    print('After Freeze')
    for k, v in model.named_parameters():
        print('\t', k, '\t', v.requires_grad)


if __name__ == '__main__':
    import torch.multiprocessing as mp

    mp.freeze_support()

    # 加载模型
    model = YOLO("yolov8s-DualConv.yaml").load("yolov8s.pt")
    #model = YOLO("yolov8s.pt")

    # 添加回调函数
    model.add_callback("on_pretrain_routine_start", freeze_model)

    # 训练模型
    results = model.train(data="datasets/plant.yaml", epochs=200, batch=8)