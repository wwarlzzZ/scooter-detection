import os
import cv2
import matplotlib.pyplot as plt


def analyze_dataset(data_path):

    print("--- Анализ структуры датасета ---")
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(data_path, split, 'images')
        if os.path.exists(img_dir):
            count = len(os.listdir(img_dir))
            print(f"В выборке [{split}]: {count} изображений")
        else:
            print(f"Предупреждение: Папка {img_dir} не найдена!")


def visualize_sample(img_path, label_path):

    if not os.path.exists(img_path) or not os.path.exists(label_path):
        print("Ошибка: Файл изображения или разметки не найден.")
        return


    image = cv2.imread(img_path)
    h, w, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = parts[0]

                x_center, y_center, bbox_w, bbox_h = map(float, parts[1:])

                xmin = int((x_center - bbox_w / 2) * w)
                ymin = int((y_center - bbox_h / 2) * h)
                xmax = int((x_center + bbox_w / 2) * w)
                ymax = int((y_center + bbox_h / 2) * h)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image, f"Class: {class_id}", (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()



if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_RAW = os.path.join(BASE_DIR, "data", "raw")

    analyze_dataset(DATA_RAW)