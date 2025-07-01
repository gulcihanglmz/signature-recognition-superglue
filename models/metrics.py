import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_comparison_matrix(json_path, output_png='signature_match_matrix.png'):
    """
    JSON dosyasındaki imza eşleşme sonuçlarını okuyup:
    - Tahmin matrisini ve ground-truth matrisini oluşturur
    - Doğruluk (accuracy) matrisini çizip kaydeder
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Aynı formatta hem dict hem list gelebilir
    pairs = data.get('pairs', data) if isinstance(data, dict) else data

    # Eşsiz imza dosyalarını topla
    unique_images = set()
    for e in pairs:
        unique_images.add(os.path.basename(e['signature1_path']))
        unique_images.add(os.path.basename(e['signature2_path']))
    unique_images = sorted(unique_images)
    n = len(unique_images)

    # Matrisleri başlat
    pred_mat = np.zeros((n, n))
    gt_mat   = np.zeros((n, n))
    acc_mat  = np.zeros((n, n))

    idx = {img: i for i, img in enumerate(unique_images)}

    # Doldur
    for e in pairs:
        i = idx[os.path.basename(e['signature1_path'])]
        j = idx[os.path.basename(e['signature2_path'])]
        p = e['prediction']
        gt = e['ground_truth_match']
        pred_mat[i, j] = pred_mat[j, i] = p
        gt_mat[i, j]   = gt_mat[j, i]   = gt
        # 2 = doğru, 1 = FP, 0 = FN
        if p == gt:
            acc_mat[i, j] = acc_mat[j, i] = 2
        elif p == 1 and gt == 0:
            acc_mat[i, j] = acc_mat[j, i] = 1
        else:
            acc_mat[i, j] = acc_mat[j, i] = 0

    # Görselleştir
    plt.figure(figsize=(20, 16))
    cmap = sns.color_palette(['#d73027', '#3182bd', '#91cf60'])  # kırmızı, mavi, yeşil
    sns.heatmap(acc_mat, xticklabels=unique_images, yticklabels=unique_images,
                cmap=cmap, square=True, annot=pred_mat, fmt='.0f', cbar=False)
    plt.title('Signature Match Accuracy Matrix\nGreen: Correct, Blue: FP, Red: FN', pad=20, size=16)
    plt.xlabel('Image 2'); plt.ylabel('Image 1')

    # İstatistikler
    total = np.sum(acc_mat >= 0) / 2
    correct = np.sum(acc_mat == 2) / 2
    fp = np.sum(acc_mat == 1) / 2
    fn = np.sum(acc_mat == 0) / 2
    stats = f'Total: {int(total)}  Correct: {int(correct)}  FP: {int(fp)}  FN: {int(fn)}  Accuracy: {correct/total:.2%}'
    plt.gcf().text(1.02, 0.5, stats, fontsize=12, va='center')

    plt.tight_layout()
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    plt.show()

    return acc_mat
