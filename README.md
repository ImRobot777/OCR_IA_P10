# Deep Learning POC: HRNetV2 + OCR vs. U-Net Baseline on Cityscapes

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Benchmark%20Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Certification-AI%20Engineer%20Project%2010-success?style=flat-square)](#)

Technology watch and Deep Learning Proof of Concept (POC) comparing HRNetV2 + OCR (Object-Contextual Representation) against a U-Net MobileNet baseline on the Cityscapes dataset for data science consultancy DataSpace.

Unlike encoder-decoder networks that compress and then reconstruct spatial dimensions, HRNet maintains high-resolution representations throughout parallel multi-resolution branches. Models are benchmarked on mIoU, Dice score, memory footprint, and inference latency, with an interactive Streamlit comparison app.

---

## Architectural Comparison

1. Classic Encoder-Decoder (U-Net):
   `Input -> Downsampling / Convolutions -> Bottleneck -> Upsampling / Skip Connections -> Mask`

2. High-Resolution Network (HRNetV2):
   Maintains a continuous high-resolution branch while progressively connecting lower-resolution branches in parallel. Repeated multi-scale fusions exchange information across resolutions, retaining precise spatial boundaries.

3. Object-Contextual Representation (OCR):
   Characterizes each pixel using the representation of its estimated object class, refining pixel-level semantic classification.

---

## Experimental Setup

* Dataset: Cityscapes (2,975 training pairs, 500 validation pairs).
* Classes: 8 macro-categories (void, flat, construction, object, nature, sky, human, vehicle).
* Image Resolution: Downscaled to 128 x 128 pixels to fit GPU memory constraints.
* Loss: Combined Focal Loss and Dice Loss.
* Batch size: 4 with early stopping and learning rate scheduling.

---

## Benchmark Summary

| Metric | Baseline (U-Net MobileNet) | Candidate POC (HRNetV2 + OCR) | Trade-Off |
| :--- | :---: | :---: | :--- |
| **Edge Precision** | Moderate | High | HRNet preserves fine structures (poles, pedestrians) |
| **mIoU** | Competitive baseline | Higher on minority classes | OCR improves context around small objects |
| **Model Size** | ~15 MB | ~120 MB | MobileNet is significantly lighter |
| **Inference Latency** | ~12 ms/frame | ~48 ms/frame | U-Net MobileNet is faster for edge devices |
| **Target Deployment** | Embedded systems / mobile | Cloud GPU inference servers | Precision vs. hardware resource trade-off |

---

## Repository Structure

```text
.
├── Livrables/
│   ├── Guitarian_Guillaume_1_plan_travail_092023.docx # Work breakdown plan
│   ├── Guitarian_Guillaume_2_notebook_092023.ipynb     # Training and evaluation notebook
│   ├── Guitarian_Guillaume_3_note_methodo_092023.docx # Methodological report
│   ├── Guitarian_Guillaume_3_note_methodo_092023.pdf  # Methodological report
│   ├── Guitarian_Guillaume_4_code_dashboard_092023.py # Streamlit comparison dashboard
│   ├── Guitarian_Guillaume_5_présentation_092023.pdf  # Defense slides
│   └── Guitarian_Guillaume_5_présentation_092023.pptx # Defense slides
├── Guitarian_Guillaume_4_code_dashboard_092023.py     # Root interactive dashboard
├── présentation.txt           # Presentation notes
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation
```

---

## Setup and Usage

### Prerequisites
* Python 3.9 or 3.10
* TensorFlow 2.x (GPU recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ImRobot777/OCR_IA_P10.git
   cd OCR_IA_P10
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install notebook tensorflow keras streamlit pillow opencv-python matplotlib seaborn pandas numpy
   ```

### Running the Comparison Dashboard
```bash
streamlit run Guitarian_Guillaume_4_code_dashboard_092023.py
```
Open `http://localhost:8501` to compare Ground Truth, U-Net, and HRNetV2+OCR segmentation masks side by side.

---

## Author & Certification

* **Student:** Guillaume Guitarian
* **Program:** OpenClassrooms AI Engineer Path (Project 10)
* **Date:** September 2023
