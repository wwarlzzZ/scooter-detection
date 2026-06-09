import os
import torch
import numpy as np
import tqdm
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt


from src.dataset.dataset import FasterRCNNDataset, collate_fn


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def box_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def evaluate_faster_rcnn(weights_path=None, output_dir=None):

    if weights_path is None:
        weights_path = os.path.join(BASE_DIR, "results", "weights", "faster_rcnn_final.pth")
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "results", "plots", "faster-rcnn")

    VAL_IMG = os.path.join(BASE_DIR, "data", "raw", "valid", "images")
    VAL_LAB = os.path.join(BASE_DIR, "data", "raw", "valid", "labels")
    CLASS_NAMES = ['bicycle', 'e-scooter', 'skateboard']

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES) + 1)

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[УСПЕХ] Веса Faster R-CNN загружены из: {weights_path}")
    else:
        print(f"[ОШИБКА] Веса не найдены по пути: {weights_path}")
        return

    model.to(device)
    model.eval()


    dataset = FasterRCNNDataset(VAL_IMG, VAL_LAB, target_size=640)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

    all_preds = []
    all_gts = []

    print("Сбор предсказаний Faster R-CNN на валидационной выборке...")
    with torch.no_grad():
        for images, targets in tqdm.tqdm(loader):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for out, target in zip(outputs, targets):
                boxes = out['boxes'].cpu().numpy()
                # Безопасное смещение классов на -1 с защитой от пустых батчей
                labels = out['labels'].cpu().numpy() - 1 if len(out['labels']) > 0 else out['labels'].cpu().numpy()
                scores = out['scores'].cpu().numpy()

                gt_boxes = target['boxes'].numpy()
                gt_labels = target['labels'].numpy() - 1 if len(target['labels']) > 0 else target['labels'].numpy()

                all_preds.append({'boxes': boxes, 'labels': labels, 'scores': scores})
                all_gts.append({'boxes': gt_boxes, 'labels': gt_labels})

    conf_thresholds = np.linspace(0.0, 0.95, 100)
    iou_threshold = 0.5

    p_curve_all = []
    r_curve_all = []
    f1_curve_all = []

    print("Вычисление метрик для построения кривых...")
    for conf_thru in conf_thresholds:
        total_tp, total_fp, total_fn = 0, 0, 0

        for pred, gt in zip(all_preds, all_gts):
            keep = pred['scores'] >= conf_thru
            p_boxes = pred['boxes'][keep]
            p_labels = pred['labels'][keep]

            g_boxes = gt['boxes']
            g_labels = gt['labels']

            matched_gt = set()
            tp, fp = 0, 0

            for p_box, p_lab in zip(p_boxes, p_labels):
                best_iou = 0
                best_gt_idx = -1
                for idx, (g_box, g_lab) in enumerate(zip(g_boxes, g_labels)):
                    if p_lab == g_lab and idx not in matched_gt and p_lab >= 0:
                        iou = box_iou(p_box, g_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx

                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn = len(g_boxes) - len(matched_gt)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        p_curve_all.append(precision)
        r_curve_all.append(recall)
        f1_curve_all.append(f1)

    f1_curve_all = np.array(f1_curve_all)
    best_f1_idx = np.argmax(f1_curve_all)
    best_f1 = f1_curve_all[best_f1_idx]
    best_conf = conf_thresholds[best_f1_idx]
    final_recall = r_curve_all[best_f1_idx]
    final_precision = p_curve_all[best_f1_idx]

    print("\n" + "=" * 50)
    print(f"ФИНАЛЬНЫЕ МЕТРИКИ FASTER R-CNN (Confidence {best_conf:.3f}):")
    print(f"Полнота (Recall):     {final_recall:.4f}")
    print(f"Точность (Precision): {final_precision:.4f}")
    print(f"F1-Score:            {best_f1:.4f}")
    print("=" * 50)

    plt.figure(figsize=(7, 6))
    plt.plot(conf_thresholds, r_curve_all, color='blue', linewidth=3,
             label=f'all classes {r_curve_all[0]:.2f} at 0.000')
    plt.xlabel('Confidence')
    plt.ylabel('Recall')
    plt.title('Recall-Confidence Curve (Faster R-CNN)')
    plt.ylim([0, 1.05])
    plt.xlim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'frcnn_Recall_curve.png'), dpi=300)
    plt.close()


    plt.figure(figsize=(7, 6))
    plt.plot(conf_thresholds, f1_curve_all, color='blue', linewidth=3,
             label=f'all classes {best_f1:.2f} at {best_conf:.3f}')
    plt.xlabel('Confidence')
    plt.ylabel('F1')
    plt.title('F1-Confidence Curve (Faster R-CNN)')
    plt.ylim([0, 1.05])
    plt.xlim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'frcnn_F1_curve.png'), dpi=300)
    plt.close()


    plt.figure(figsize=(7, 6))
    plt.plot(r_curve_all, p_curve_all, color='blue', linewidth=3, label='Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Faster R-CNN)')
    plt.ylim([0, 1.05])
    plt.xlim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'frcnn_PR_curve.png'), dpi=300)
    plt.close()

    print(f"\n[ГОТОВО] Все графики успешно сохранены в папку: {output_dir}")


if __name__ == '__main__':
    evaluate_faster_rcnn()