# Qwave
QWave is a Quantum Machine Learning (QML) application that detects digital watermarks in images using variational quantum circuits and amplitude embedding. It combines classical image processing with quantum intelligence to deliver fast, accurate, and visually intuitive watermark detection.

# 🚀 Quantum Watermark Detector

**A Quantum Machine Learning (QML) Application for Detecting Digital Watermarks in Images Using Quantum Computing Principles**

## 📌 Project Overview

**Name:** Quantum Watermark Detector  
**Type:** Quantum Machine Learning (QML) Application  
**Goal:** Detect digital watermarks in images using quantum computing principles  
**Approach:** Variational quantum circuits with amplitude embedding to classify images as *“watermarked”* or *“clean”*

This project provides a **complete end-to-end implementation** that combines:

- 🧠 Classical Image Processing  
- ⚛️ Quantum Machine Learning  
- 💻 Professional GUI Interface  
- 📊 Real-world Dataset (CIFAR-100)  
- 🧩 Model Training + Live Detection  

Perfect for **academic projects**, **course assignments**, or **quantum computing portfolios**.

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

## 5. 🗃️ Folder Structure
qml-watermark-detector/
├── venv/ # Virtual environment
├── data/ # Data storage
│ ├── raw/ # Raw CIFAR-100 data
│ ├── processed/ # Cached feature vectors
│ └── results/ # Training outputs, plots, model weights
├── src/ # Source code
│ ├── data_generator.py # Synthetic data generator (backup)
│ ├── quantum_model.py # Full quantum model (training)
│ ├── quantum_predictor.py # Lightweight model for GUI prediction
│ ├── trainer.py # Training logic
│ ├── evaluator.py # Evaluation metrics
│ └── cifar_watermark_processor.py # CIFAR-100 + watermark embedding
├── main.py # Train the model (run first)
├── gui_detector.py # GUI for live watermark detection
├── requirements.txt # Dependencies list
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

🎯 Key Points
✅ Fully working Quantum ML project
✅ Complete training + evaluation pipeline
✅ Interactive GUI with visualizations
✅ Real-world CIFAR-100 dataset
✅ True quantum circuit implementation
✅ Amplitude embedding + variational circuits
✅ Feature importance heatmaps
✅ Clean, documented, and modular codebase
✅ Ready for presentation or portfolio submission

🛠️ Customization Options:
🔧 Change watermark pattern → src/cifar_watermark_processor.py
📈 Adjust number of training samples → main.py
⚛️ Modify quantum circuit layers → src/quantum_model.py
🧮 Add new feature extraction methods (e.g., frequency domain features)
📂 Extend compatibility to other datasets (e.g., ImageNet, COCO)

