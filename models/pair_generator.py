import os
import json
from tqdm import tqdm
from itertools import combinations
from models.matcher_utils import match_signatures_superglue

def generate_signature_pairs_and_analyze_superglue(root_folder, output_json_path,
                                                   matcher,
                                                   device='cuda' if torch.cuda.is_available() else 'cpu',
                                                   ratio_thresh=0.2,
                                                   conf_thresh=0.2):
    folder_images = {}

    # Alt klasörlerden görselleri topla
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            images = [f for f in os.listdir(folder_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if images:
                folder_images[folder] = [os.path.join(folder_path, img) for img in sorted(images)]

    results = []
    processed_pairs = set()
    folders = sorted(folder_images.keys())

    with tqdm(desc="SuperGlue Signature Matching") as pbar:
        # Pozitif eşleşmeler (aynı klasör içinde)
        for folder in folders:
            folder_pairs = list(combinations(folder_images[folder], 2))
            for img1, img2 in folder_pairs:
                pair_key = tuple(sorted([img1, img2]))
                if pair_key not in processed_pairs:
                    try:
                        pred, ratio, conf = match_signatures_superglue(img1, img2, matcher,
                                                                       device=device,
                                                                       ratio_thresh=ratio_thresh,
                                                                       conf_thresh=conf_thresh)
                        results.append({
                            'signature1_path': img1,
                            'signature2_path': img2,
                            'ground_truth_match': 1,
                            'prediction': int(pred),
                            'match_ratio': float(ratio),
                            'confidence_score': float(conf)
                        })
                    except Exception as e:
                        print(f"Error processing {img1} <-> {img2}: {str(e)}")
                    processed_pairs.add(pair_key)
                    pbar.update(1)

        # Negatif eşleşmeler (farklı klasörler)
        for i, folder1 in enumerate(folders):
            for folder2 in folders[i + 1:]:
                for img1 in folder_images[folder1]:
                    for img2 in folder_images[folder2]:
                        pair_key = tuple(sorted([img1, img2]))
                        if pair_key not in processed_pairs:
                            try:
                                pred, ratio, conf = match_signatures_superglue(img1, img2, matcher,
                                                                               device=device,
                                                                               ratio_thresh=ratio_thresh,
                                                                               conf_thresh=conf_thresh)
                                results.append({
                                    'signature1_path': img1,
                                    'signature2_path': img2,
                                    'ground_truth_match': 0,
                                    'prediction': int(pred),
                                    'match_ratio': float(ratio),
                                    'confidence_score': float(conf)
                                })
                            except Exception as e:
                                print(f"Error processing {img1} <-> {img2}: {str(e)}")
                            processed_pairs.add(pair_key)
                            pbar.update(1)

    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ JSON saved to: {output_json_path}")
    return results
