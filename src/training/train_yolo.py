from ultralytics import YOLO
import torch

def train():
    model = YOLO('yolo26n.pt')

    results = model.train(
        data='../../configs/data.yaml',
        epochs=50,
        imgsz=640,
        batch=32,
        amp=True,
        device=0,
        workers=4
    )

if __name__ == '__main__':
    train()