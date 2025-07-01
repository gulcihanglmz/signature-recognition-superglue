import torch
from models.matching import Matching

# SuperPoint ve SuperGlue için yapılandırma ayarları
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': 'indoor',  # indoor: kapalı ortamlar için optimize edilmiş
        'sinkhorn_iterations': 20,
        'match_threshold': 0.15
    }
}

# CUDA varsa GPU kullan, yoksa CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Eşleştirici model (SuperPoint + SuperGlue)
matcher = Matching(config).eval().to(device)