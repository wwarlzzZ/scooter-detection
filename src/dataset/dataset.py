import os
import torch
from PIL import Image
import torchvision
from torch.utils.data import Dataset


def yolo_to_pascal(yolo_box, target_w, target_h):

    cls_id, x_c, y_c, w, h = yolo_box

    x_c, w = x_c * target_w, w * target_w
    y_c, h = y_c * target_h, h * target_h

    xmin = int(x_c - (w / 2))
    ymin = int(y_c - (h / 2))
    xmax = int(x_c + (w / 2))
    ymax = int(y_c + (h / 2))

    xmin = max(0, min(xmin, target_w - 1))
    ymin = max(0, min(ymin, target_h - 1))
    xmax = max(xmin + 1, min(xmax, target_w))
    ymax = max(ymin + 1, min(ymax, target_h))

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

        img_tensor = torchvision.transforms.functional.to_tensor(img)

        label_name = os.path.splitext(self.img_names[idx])[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue

                    data = list(map(float, line.split()))
                    if len(data) != 5:
                        continue

                    cls_id = int(data[0])
                    _, box = yolo_to_pascal(data, self.target_size, self.target_size)
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