from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_yolo_model(config_path=None, epochs=50, batch=16):
    if config_path is None:
        config_path = os.path.join(BASE_DIR, "data", "raw", "data.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Конфигурационный файл не найден по пути: {config_path}")

    model = YOLO('yolov8n.pt')

    print(f"--- Запуск обучения YOLO на {epochs} эпох ---")

    project_dir = os.path.join(BASE_DIR, "results")

    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=0,
        workers=0,
        amp=True,
        project=project_dir,
        name="yolo_train"
    )
    return results


if __name__ == '__main__':

    DATA_YAML = os.path.join(BASE_DIR, "data", "raw", "data.yaml")
    train_yolo_model(config_path=DATA_YAML, epochs=50, batch=16)