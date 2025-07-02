from superglue_config import matcher, device
from models.pair_generator import generate_signature_pairs_and_analyze_superglue
from models.visualizer import visualize_signature_match
from models.metrics import create_comparison_matrix

if __name__ == '__main__':
    root_folder = r"D:\NeviTecth\signature_dataset\signature_dataset\original"
    output_json = r"D:\NeviTecth\signature_dataset\signature_dataset\original_results.json"

    results = generate_signature_pairs_and_analyze_superglue(
        root_folder, output_json, matcher, device=device
    )

    create_comparison_matrix(output_json, output_png='signature_accuracy_matrix.png')

    img1 = f"{root_folder}/1/color_10.png"
    img2 = f"{root_folder}/1/color_14.png"
    visualize_signature_match(img1, img2, matcher, device=device)
