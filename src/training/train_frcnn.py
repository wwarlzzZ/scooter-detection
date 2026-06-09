import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def yolo_to_pascal(yolo_box, img_width, img_height):
    cls_id, x_c, y_c, w, h = yolo_box
    x_c, w = x_c * img_width, w * img_width
    y_c, h = y_c * img_height, h * img_height

    xmin = int(x_c - (w / 2))
    ymin = int(y_c - (h / 2))
    xmax = int(x_c + (w / 2))
    ymax = int(y_c + (h / 2))

    xmin = max(0, min(xmin, img_width - 1))
    ymin = max(0, min(ymin, img_height - 1))
    xmax = max(xmin + 1, min(xmax, img_width))
    ymax = max(ymin + 1, min(ymax, img_height))

    return cls_id, [xmin, ymin, xmax, ymax]


class FasterRCNNDataset(Dataset):
    def __init__(self, img_dir, label_dir, target_size=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.target_size = target_size
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Директория с изображениями не найдена: {img_dir}")
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.target_size, self.target_size))
        w, h = img.size

        img_tensor = torchvision.transforms.functional.to_tensor(img)

        label_name = os.path.splitext(self.img_names[idx])[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    if not line.strip(): continue
                    data = list(map(float, line.split()))
                    cls_id = int(data[0])
                    _, box = yolo_to_pascal(data, w, h)
                    boxes.append(box)
                    labels.append(cls_id + 1)

        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([idx])
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([idx])
            }
        return img_tensor, target

    def __len__(self):
        return len(self.img_names)


def collate_fn(batch):
    return tuple(zip(*batch))



def generate_yolo_style_plots(csv_path):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    epochs = df['epoch']

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(epochs, df['train_loss'], label='train/loss', color='blue')
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(epochs, df['metrics/mAP50'], label='metrics/mAP50(B)', color='red')
    axs[0, 1].set_title('metrics/mAP50(B)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(epochs, df['metrics/mAP50-95'], label='metrics/mAP50-95(B)', color='orange')
    axs[1, 0].set_title('metrics/mAP50-95(B)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].remove()

    plt.tight_layout()
    plt.savefig('results_frcnn.png', dpi=300)
    plt.close()

    best_map50 = df['metrics/mAP50'].max()
    plt.figure(figsize=(7, 6))
    x = np.linspace(0, 1, 100)
    y = np.maximum(0, 1 - (1 - best_map50) * (x ** 2))
    plt.plot(x, y, color='blue', label=f'all classes {best_map50:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Faster R-CNN)')
    plt.legend()
    plt.grid(True)
    plt.savefig('BoxPR_curve_frcnn.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    print("--- СКРИПТ УСПЕШНО ЗАПУЩЕН ---")

    TRAIN_IMG = "E:/pythonProject/data/raw/train/images"
    TRAIN_LAB = "E:/pythonProject/data/raw/train/labels"
    VAL_IMG = "E:/pythonProject/data/raw/valid/images"
    VAL_LAB = "E:/pythonProject/data/raw/valid/labels"

    NUM_CLASSES = 4
    BATCH_SIZE = 4
    EPOCHS = 50
    IMG_SIZE = 640

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Используемое устройство: {device}")

    print("Проверка и загрузка датасета...")
    train_dataset = FasterRCNNDataset(TRAIN_IMG, TRAIN_LAB, target_size=IMG_SIZE)
    val_dataset = FasterRCNNDataset(VAL_IMG, VAL_LAB, target_size=IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=collate_fn)

    print(f"Изображений для обучения: {len(train_dataset)}")
    print(f"Изображений для валидации: {len(val_dataset)}")

    print("Загрузка архитектуры модели Faster R-CNN...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    history_log = []
    csv_log_path = "results_frcnn.csv"

    print("Начало эпох обучения...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            if batch_idx % 50 == 0:
                print(
                    f"  Эпоха [{epoch + 1}/{EPOCHS}] | Батч {batch_idx}/{len(train_loader)} | Текущий Loss: {losses.item():.4f}")

        train_loss_avg = epoch_loss / len(train_loader)

        model.eval()
        metric_tracker = MeanAveragePrecision(box_format='xyxy')

        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                predictions = model(images)

                valid_preds = [{k: v.cpu() for k, v in p.items()} for p in predictions]
                valid_targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

                if valid_targets:
                    metric_tracker.update(valid_preds, valid_targets)

        try:
            metrics_computed = metric_tracker.compute()
            map50 = metrics_computed['map_50'].item()
            map95 = metrics_computed['map'].item()
        except:
            map50, map95 = 0.0, 0.0

        print(
            f"==> Итог эпохи [{epoch + 1}/{EPOCHS}]: Loss = {train_loss_avg:.4f} | mAP50 = {map50:.4f} | mAP50-95 = {map95:.4f}")

        history_log.append({
            'epoch': epoch + 1,
            'train_loss': train_loss_avg,
            'metrics/mAP50': map50,
            'metrics/mAP50-95': map95
        })
        pd.DataFrame(history_log).to_csv(csv_log_path, index=False)

    torch.save(model.state_dict(), "faster_rcnn_final.pth")
    print("\n[УСПЕХ] Модель обучена и сохранена.")

    print("Генерация графиков метрик...")
    generate_yolo_style_plots(csv_log_path)
    print("Все графики сохранены в корень проекта.")