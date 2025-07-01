import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.image_utils import resize_with_aspect_ratio, pad_to_same_height, rotate_image
from models.matcher_utils import best_rotated_match

def visualize_signature_match(img_path1, img_path2, matcher, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    İki imza görüntüsünü yan yana çizip eşleşen anahtar noktaları gösterir,
    ayrıca dosya adlarını ve eşleşme oranı ile en iyi döndürme açısını başlıkta yazar.
    """

    # Görselleri oku
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError("One of the input images could not be loaded.")

    # Yeniden boyutlandır
    img1 = resize_with_aspect_ratio(img1)
    img2 = resize_with_aspect_ratio(img2)

    # Dosya adlarını al
    name1 = os.path.splitext(os.path.basename(img_path1))[0]
    name2 = os.path.splitext(os.path.basename(img_path2))[0]

    # En iyi açıyla eşleşmeyi bul
    _, match_ratio, _, best_angle = best_rotated_match(img1, img2, matcher, device)

    # İkinci görseli en iyi açıyla döndür
    rotated_img2 = rotate_image(img2, best_angle)

    # Yükseklikleri eşitle
    img1, rotated_img2 = pad_to_same_height(img1, rotated_img2)

    # Tensörlere çevir ve anahtar noktaları eşle
    img1_tensor = torch.from_numpy(img1 / 255.).float()[None, None].to(device)
    img2_tensor = torch.from_numpy(rotated_img2 / 255.).float()[None, None].to(device)
    pred = matcher({'image0': img1_tensor, 'image1': img2_tensor})
    kpts0 = pred['keypoints0'][0].detach().cpu().numpy()
    kpts1 = pred['keypoints1'][0].detach().cpu().numpy()
    matches = pred['matches0'][0].detach().cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    # Yan yana birleştir
    match_img = cv2.hconcat([
        cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(rotated_img2, cv2.COLOR_GRAY2BGR)
    ])

    # Dosya adlarını yaz
    cv2.putText(match_img, name1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.putText(match_img, name2, (img1.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Eşleşme çizgilerini çiz
    for pt1, pt2 in zip(mkpts0, mkpts1):
        pt1_i = tuple(np.round(pt1).astype(int))
        pt2_i = tuple(np.round(pt2).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(match_img, pt1_i, pt2_i, (0, 255, 0), 1)

    # Başlık ve gösterim
    title = f"Matched" if match_ratio >= 0.2 else "Not Matched"
    plt.figure(figsize=(20, 8))
    plt.imshow(match_img[..., ::-1])
    plt.title(f"{title} - Match ratio: {match_ratio:.2%} | Best rotation: {best_angle}°", fontsize=16)
    plt.axis('off')
    plt.show()

    return match_ratio
