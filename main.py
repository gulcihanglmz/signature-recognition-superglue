from superglue_config import matcher, device
from models.pair_generator import generate_signature_pairs_and_analyze_superglue
from models.visualizer import visualize_signature_match
from models.metrics import create_comparison_matrix

if __name__ == '__main__':
    # 1. Dataset yolunu ve JSON çıkışını tanımla
    root_folder = "/path/to/signature_dataset/original"
    output_json = "/path/to/signature_dataset/original_results.json"

    # 2. Tüm imza çiftlerini karşılaştır ve JSON'a kaydet
    results = generate_signature_pairs_and_analyze_superglue(
        root_folder, output_json, matcher, device=device
    )

    # 3. Eşleşme matrisini oluştur ve görselleştir
    create_comparison_matrix(output_json, output_png='signature_accuracy_matrix.png')

    # 4. Örnek bir çift için görselleştirme
    img1 = f"{root_folder}/1/color_10.png"
    img2 = f"{root_folder}/1/color_14.png"
    visualize_signature_match(img1, img2, matcher, device=device)
