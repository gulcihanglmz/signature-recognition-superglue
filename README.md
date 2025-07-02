# Signature Recognition with SuperGlue

A Python toolkit for pairwise signature matching using [SuperPoint](https://arxiv.org/abs/1712.07629) + [SuperGlue](https://arxiv.org/abs/1911.11763).  
It generates a JSON of match predictions for all signature pairs in your dataset, then visualizes results for inspection.

---

## 📂 Repository Structure

```
.
├── .venv/                            # Python virtual environment
├── models/                          # Utility modules
│   ├── **init**.py
│   ├── image\_utils.py               # resize, rotate, pad functions
│   ├── matcher\_utils.py             # best\_rotated\_match wrapper
│   ├── visualizer.py                # visualize\_signature\_match
│   ├── pair\_generator.py            # dataset traversal & JSON export
│   ├── metrics.py                   # comparison‐matrix heatmap
├── SuperGluePretrainedNetwork/      # Official SuperGlue code & weights
│   └── models/
│       ├── matching.py
│       ├── superpoint.py
│       ├── superglue.py
│       └── utils.py
├── superglue\_config.py              # superpoint + superglue init & config
├── main.py                          # end-to-end script
├── requirements.txt
└── README.md

````

---

## 🚀 Installation

1. **Clone** this repo and `cd` into it:
   ```bash
   git clone https://github.com/yourusername/signature-recognition-superglue.git
   cd signature-recognition-superglue
````

2. **Set up** a virtual environment & install dependencies:

   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   ```
3. **Download** the SuperGlue pretrained code and weights, placing them under `SuperGluePretrainedNetwork/` (as shown above).
   You can grab the official implementation from:
   [https://github.com/magicleap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)

---
## 🛠️ Usage

1. **Generate matches JSON**
   Compares every pair of signatures in `root_folder`, saves results to `output_json`:

   ```bash
   python main.py --root_folder /path/to/signature_dataset/original \
                  --output_json /path/to/output/original_results.json
   ```
2. **Visualize a single pair**

   ```python
   from visualizer import visualize_signature_match
   from superglue_config import matcher, device

   img1 = "/path/to/signature_dataset/img1"
   img2 = "/path/to/signature_dataset/img2"

   visualize_signature_match(img1, img2, matcher, device=device)
   ```
3. **Batch-visualize from JSON**

   ```python
   import json
   from visualizer import visualize_signature_match
   from superglue_config import matcher, device

   with open("original_results.json") as f:
       data = json.load(f)

   for entry in data:
       visualize_signature_match(
           entry["signature1_path"],
           entry["signature2_path"],
           matcher,
           device=device
       )
   ```

---

## 🎯 What’s Inside Each Module?

* **`superglue_config.py`**

  * Initializes SuperPoint & SuperGlue with your chosen hyper‐parameters (`indoor` vs. `outdoor` weights, thresholds, etc.).

* **`image_utils.py`**

  * `resize_with_aspect_ratio`, `rotate_image`, `pad_to_same_height` – image pre‐ and post‐processing helpers.

* **`matcher_utils.py`**

  * `best_rotated_match` – tries multiple rotations (0°, 90°, 180°, 270°), picks the best match angle/ratio.

* **`pair_generator.py`**

  * Walks through `root_folder` subdirectories, generates positive (same-folder) and negative (cross-folder) pairs, runs the matcher, dumps JSON.

* **`visualizer.py`**

  * Draws keypoint matches between two images side-by-side, annotates file names, match ratio & best rotation.

* **`metrics.py`**

  * Builds a heatmap of match accuracy across all pairs (useful for overall performance analysis).

* **`main.py`**

  * A single script to tie everything together end-to-end.

---

## 📝 References

* **SuperGlue Pretrained Network** (Matching backbone):
  [https://github.com/magicleap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)
* **SuperPoint & SuperGlue papers** for algorithmic details:

  * DeTone, Malisiewicz & Rabinovich, “SuperPoint: Self-Supervised Interest Point Detection and Description”, ECCV 2018.
  * Sarlin et al., “SuperGlue: Learning Feature Matching with Graph Neural Networks”, CVPR 2020.

---

## 🤝 Contributing

Feel free to open issues or pull requests! Any enhancement—e.g. adding finer rotation steps, retraining on signature data, improving visualization—is welcome.

---

*Happy matching!* 🚀
