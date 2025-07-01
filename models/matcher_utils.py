import torch
import numpy as np
from models.image_utils import rotate_image

def best_rotated_match(img1, img2, matcher, device, ratio_thresh=0.2, conf_thresh=0.2):
    best_pred = 0
    best_ratio = 0
    best_conf = 0
    best_angle = 0

    # Farklı dönüş açılarıyla eşleşme denemesi
    for angle in [0, 90, 180, 270]:
        rotated_img2 = rotate_image(img2, angle)

        img1_tensor = torch.from_numpy(img1 / 255.).float()[None, None].to(device)
        img2_tensor = torch.from_numpy(rotated_img2 / 255.).float()[None, None].to(device)

        pred = matcher({'image0': img1_tensor, 'image1': img2_tensor})
        matches = pred['matches0'][0].detach().cpu().numpy()
        kpts0 = pred['keypoints0'][0].detach().cpu().numpy()
        scores = pred['matching_scores0'][0].detach().cpu().numpy()

        valid = matches > -1
        ratio = np.sum(valid) / max(len(kpts0), 1)
        conf = scores[valid].mean() if valid.any() else 0
        match = 1 if ratio >= ratio_thresh and conf >= conf_thresh else 0

        if ratio > best_ratio:
            best_pred = match
            best_ratio = ratio
            best_conf = conf
            best_angle = angle

    print(f"📐 Best matching angle: {best_angle}°")
    return best_pred, best_ratio, best_conf, best_angle


def match_signatures_superglue(img_path1, img_path2, matcher,
                               device='cuda' if torch.cuda.is_available() else 'cpu',
                               ratio_thresh=0.2, conf_thresh=0.2):
    import cv2
    from models.image_utils import resize_with_aspect_ratio

    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError("One of the images could not be loaded.")

    img1 = resize_with_aspect_ratio(img1)
    img2 = resize_with_aspect_ratio(img2)

    prediction, match_ratio, avg_conf, _ = best_rotated_match(
        img1, img2, matcher, device, ratio_thresh=ratio_thresh, conf_thresh=conf_thresh
    )

    return prediction, match_ratio, avg_conf
