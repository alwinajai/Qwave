# 🚀 Qwave (Quantum Watermark Detector )
QWave is a Quantum Machine Learning (QML) application that detects digital watermarks in images using variational quantum circuits and amplitude embedding. It combines classical image processing with quantum intelligence to deliver fast, accurate, and visually intuitive watermark detection.

## 📌 Project Overview
QWave is a cutting-edge Quantum Machine Learning (QML) application designed to detect hidden digital watermarks in images using principles of quantum computing. Unlike classical watermark detection methods, QWave leverages variational quantum circuits and amplitude embedding to classify images as “watermarked” or “clean” with quantum advantage.
This project bridges the gap between theoretical quantum computing and practical computer vision applications — demonstrating how quantum algorithms can be applied to real-world problems like content authentication, copyright protection, and digital forensics.

## 🧩 Core Components
### 1. Data Processing
- Uses **CIFAR-100** dataset (100 classes, 32×32 RGB images).  
- Algorithmically embeds **4×4 binary watermark patterns** in half the images.  
- Extracts **16 statistical features** per image (mean, std, block means, etc.).  
- Normalizes data for **quantum amplitude embedding**.

### 2. Quantum Model
- **4-qubit circuit** efficiently encodes 16 features via amplitude embedding.  
- **3-layer variational quantum circuit** with RX, RY, RZ + CNOT gates.  
- **Pauli-Z measurement** converts results to probabilities (0–1).  
- **Binary classification:**  
  - `0` → Clean image  
  - `1` → Watermarked image  

### 3. Training Pipeline
- Uses **gradient descent with numerical gradients**.  
- Trains for **35 epochs**, optimized for speed and performance.  
- Saves model weights and metrics under `data/results/`.  
- Generates **training history plots** and **evaluation metrics**.

### 4. GUI Application
- Modern, professional interface with **dark/light themes**.  
- Select any image file (`.png`, `.jpg`, `.bmp`, `.tiff`, etc.).  
- Real-time analysis with progress bar.  
- **Feature importance visualization** via heatmap.  
- Clear output:  
  - 🟢 “Watermark Detected!”  
  - 🔵 “No Watermark Detected.”


## 5. 🗃️ Folder Structure ( View the folder structure properly in the edit mode of Readme.md file )
qml-watermark-detector/
QWave/
├── venv/ # Virtual environment (optional, not uploaded to GitHub)
│
├── data/ # Dataset and experiment outputs
│ ├── raw/ # Original CIFAR-100 dataset
│ ├── processed/ # Preprocessed feature vectors (cached)
│ └── results/ # Training outputs, model weights, logs, and plots
│
├── src/ # Source code (main logic)
│ ├── preprocessing/ # Image + watermark data handling
│ │ └── cifar_watermark_processor.py # Embeds and processes watermarks in CIFAR images
│ │
│ ├── model/ # Quantum model architecture
│ │ ├── quantum_model.py # Full variational quantum circuit (training model)
│ │ └── quantum_predictor.py # Lightweight circuit for GUI-based predictions
│ │
│ ├── training/ # Training and evaluation pipeline
│ │ ├── trainer.py # Model training logic and optimization
│ │ └── evaluator.py # Model evaluation and metric generation
│ │
│ ├── utils/ # Helper modules (feature extraction, normalization, etc.)
│ │ └── data_generator.py # Synthetic data generator (backup or testing)
│ │
│ └── gui/ # Graphical User Interface
│ └── gui_detector.py # Live detection interface with visualization
│
├── main.py # Entry point — trains the QWave model
│
├── requirements.txt # List of dependencies
│
└── README.md # Project documentation

##⚙️ Installation & Setup
###1. Clone the Repository
```bash
git clone https://github.com/yourusername/qml-watermark-detector.git
cd qml-watermark-detector

##2. Create Virtual Environment
python3.10 -m venv venv
# Activate the environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Train the Model (Run Once)
python main.py

This will:
Download CIFAR-100
Train the quantum model
Save results, plots, and model weights in data/results/

5. Run the GUI (for Live Detection)
python gui_detector.py

🛠️ Customization Options:
🔧 Change watermark pattern → src/cifar_watermark_processor.py
📈 Adjust number of training samples → main.py
⚛️ Modify quantum circuit layers → src/quantum_model.py
🧮 Add new feature extraction methods (e.g., frequency domain features)
📂 Extend compatibility to other datasets (e.g., ImageNet, COCO)

