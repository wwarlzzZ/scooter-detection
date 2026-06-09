import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision


from src.dataset.dataset import FasterRCNNDataset, collate_fn
from src.utils.utils import generate_yolo_style_plots


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_frcnn_model(epochs=50, batch_size=4):
    print("--- ЗАПУСК ОБУЧЕНИЯ FASTER R-CNN ---")

    TRAIN_IMG = os.path.join(BASE_DIR, "data", "raw", "train", "images")
    TRAIN_LAB = os.path.join(BASE_DIR, "data", "raw", "train", "labels")
    VAL_IMG = os.path.join(BASE_DIR, "data", "raw", "valid", "images")
    VAL_LAB = os.path.join(BASE_DIR, "data", "raw", "valid", "labels")

    NUM_CLASSES = 4
    IMG_SIZE = 640

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Используемое устройство: {device}")

    train_dataset = FasterRCNNDataset(TRAIN_IMG, TRAIN_LAB, target_size=IMG_SIZE)
    val_dataset = FasterRCNNDataset(VAL_IMG, VAL_LAB, target_size=IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, collate_fn=collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    history_log = []
    log_dir = os.path.join(BASE_DIR, "results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_log_path = os.path.join(log_dir, "results_frcnn.csv")

    print("Начало эпох обучения...")
    for epoch in range(epochs):
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
                print(f"  Эпоха [{epoch + 1}/{epochs}] | Батч {batch_idx}/{len(train_loader)} | Loss: {losses.item():.4f}")

        train_loss_avg = epoch_loss / len(train_loader)

        model.eval()
        metric_tracker = MeanAveragePrecision(box_format='xyxy')

        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                predictions = model(images)

                valid_preds = [{k: v.detach().cpu() for k, v in p.items()} for p in predictions]
                valid_targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

                if valid_targets:
                    metric_tracker.update(valid_preds, valid_targets)

        try:
            metrics_computed = metric_tracker.compute()
            map50 = metrics_computed['map_50'].item()
            map95 = metrics_computed['map'].item()
        except Exception as e:
            print(f"Ошибка при расчете mAP: {e}")
            map50, map95 = 0.0, 0.0

        print(f"==> Итог эпохи [{epoch + 1}/{epochs}]: Loss = {train_loss_avg:.4f} | mAP50 = {map50:.4f}")

        history_log.append({
            'epoch': epoch + 1,
            'train_loss': train_loss_avg,
            'metrics/mAP50': map50,
            'metrics/mAP50-95': map95
        })
        pd.DataFrame(history_log).to_csv(csv_log_path, index=False)


    weights_dir = os.path.join(BASE_DIR, "results", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(weights_dir, "faster_rcnn_final.pth"))
    print("\n[УСПЕХ] Финальные веса Faster R-CNN успешно сохранены.")


    print("Генерация графиков метрик...")
    plots_dir = os.path.join(BASE_DIR, "results", "plots", "faster-rcnn")
    generate_yolo_style_plots(csv_log_path, output_dir=plots_dir)

if __name__ == '__main__':

    train_frcnn_model(epochs=5, batch_size=4)