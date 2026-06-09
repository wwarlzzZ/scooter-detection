import os
import pandas as pd
import matplotlib.pyplot as plt


def generate_yolo_style_plots(csv_path, output_dir="results/plots/faster-rcnn"):

    if not os.path.exists(csv_path):
        print(f"[ПРЕДУПРЕЖДЕНИЕ] CSV файл не найден для графиков: {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    epochs = df['epoch']

    fig = plt.figure(figsize=(12, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(epochs, df['train_loss'], label='train/loss', color='blue', linewidth=2)
    ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()


    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(epochs, df['metrics/mAP50'], label='metrics/mAP50(B)', color='red', linewidth=2)
    ax2.set_title('metrics/mAP50 (B)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP50')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()


    ax3 = plt.subplot(2, 1, 2)
    ax3.plot(epochs, df['metrics/mAP50-95'], label='metrics/mAP50-95(B)', color='orange', linewidth=2)
    ax3.set_title('metrics/mAP50-95 (B)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('mAP50-95')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()

    plt.tight_layout()


    output_plot_path = os.path.join(output_dir, 'results_frcnn_training.png')
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

    print(f"[УСПЕХ] Графики процесса обучения сохранены в: {output_plot_path}")


if __name__ == '__main__':
    pass