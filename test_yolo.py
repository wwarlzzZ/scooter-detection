import os
from ultralytics import YOLO

if __name__ == '__main__':

    model_path = "E:/pythonProject/results/weights/bestyolo26s.pt"

    if not os.path.exists(model_path):
        print(f"❌ Веса не найдены по пути: {model_path}")
        print("Проверь папку runs/detect/ и посмотри точное название папки обучения!")
    else:

        model = YOLO(model_path)

        source_path = "E:/pythonProject/data/raw/test/images"

        print(" Запускаем детекцию на тестовой выборке")


        results = model.predict(
            source=source_path,
            conf=0.35,
            save=True,
            save_txt=False,
            device=0
        )

        print("🎯 Готово! Результаты сохранены в папку runs/detect/predict/")