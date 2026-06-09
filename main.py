import argparse
import sys
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.training.train_yolo import train_yolo_model
from src.training.train_frcnn import train_frcnn_model
from src.evaluation.metrics import evaluate_faster_rcnn


def main():
    parser = argparse.ArgumentParser(description="Репозиторий детекции электросамокатов в закрытой зоне")

    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="Режим: 'train' для обучения или 'eval' для валидации")

    parser.add_argument("--model", type=str, default="yolo", choices=["yolo", "faster_rcnn"],
                        help="Архитектура модели: 'yolo' или 'faster_rcnn'")

    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох обучения")
    parser.add_argument("--batch", type=int, default=16, help="Размер батча для обучения")

    args = parser.parse_args()


    DATA_YAML_PATH = os.path.join(BASE_DIR, "data", "raw", "data.yaml")
    FRCNN_WEIGHTS = os.path.join(BASE_DIR, "results", "weights", "faster_rcnn_final.pth")

    if args.mode == "train":
        if args.model == "yolo":
            print(f"[ИНФО] Запуск обучения YOLO. Конфиг: {DATA_YAML_PATH}")
            train_yolo_model(config_path=DATA_YAML_PATH, epochs=args.epochs, batch=args.batch)

        elif args.model == "faster_rcnn":

            frcnn_batch = args.batch if args.batch <= 4 else 4
            print(f"[ИНФО] Запуск обучения Faster R-CNN. Ограничение батча: {frcnn_batch}")
            train_frcnn_model(epochs=args.epochs, batch_size=frcnn_batch)

    elif args.mode == "eval":
        if args.model == "faster_rcnn":
            print(f"[ИНФО] Запуск расчета честных метрик для Faster R-CNN...")
            evaluate_faster_rcnn(weights_path=FRCNN_WEIGHTS)

        elif args.model == "yolo":
            print("\n" + "=" * 60)
            print("[ИНФО] Для валидации YOLO используйте стандартный движок Ultralytics:")
            print(f"yolo detect val model=results/weights/yolov8_best.pt data={DATA_YAML_PATH}")
            print("=" * 60)


if __name__ == "__main__":
    main()