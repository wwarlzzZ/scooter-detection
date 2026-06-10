import os
import torch
import cv2
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


def load_model(weights_path, num_classes=2):

    model = fasterrcnn_resnet50_fpn(weights=None)


    in_features = model.roi_heads.box_predictor.cls_score.in_features


    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    checkpoint = torch.load(weights_path, map_location=device)


    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, device


if __name__ == '__main__':

    WEIGHTS_PATH = "E:/pythonProject/results/weights/faster_rcnn_final.pth"
    TEST_IMAGES_DIR = "E:/pythonProject/data/raw/test/images"
    OUTPUT_DIR = "E:/pythonProject/runs/detect/predict_faster"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(WEIGHTS_PATH):
        print(f" Веса Faster R-CNN не найдены: {WEIGHTS_PATH}")
    else:

        model, device = load_model(WEIGHTS_PATH, num_classes=4)
        print(" Запуск инференса Faster R-CNN ")

        CONF_THRESHOLD = 0.35

        images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

        with torch.no_grad():
            for img_name in images:
                img_path = os.path.join(TEST_IMAGES_DIR, img_name)
                orig_image = cv2.imread(img_path)

                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                image_tensor = F.to_tensor(image).unsqueeze(0).to(device)


                predictions = model(image_tensor)[0]


                boxes = predictions['boxes'].cpu().numpy()
                scores = predictions['scores'].cpu().numpy()
                labels = predictions['labels'].cpu().numpy()


                for box, score, label in zip(boxes, scores, labels):
                    if score >= CONF_THRESHOLD:
                        xmin, ymin, xmax, ymax = box.astype(int)


                        cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

                        text = f"scooter: {score:.2f}"
                        cv2.putText(orig_image, text, (xmin, ymin - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), orig_image)

        print(f"🎯 Проверка завершена! Ищи результаты здесь: {OUTPUT_DIR}")