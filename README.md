
# 🍎 Fruit Quality Assessment & Defect Detection System

An AI-powered computer vision system for fruit quality classification, visual defect assessment, and explainable predictions using deep learning.

## ✨ Features
- Multi-fruit quality classification
- MobileNetV2 & DenseNet-121 transfer learning models
- Grad-CAM visual explainability
- Live camera-based inspection
- Batch image processing
- Image stress testing (brightness, noise, blur, etc.)
- Interactive Streamlit analytics dashboard
- Optional Arduino/Raspberry Pi integration

## 🧠 Models
- **MobileNetV2** — Lightweight, efficient image classification.
- **DenseNet-121** — Deep feature extraction and classification.

## 🍇 Supported Fruits
Apples, Bananas, Oranges, Guavas, Pomegranates, and Mangoes.

## 📊 Dataset Split
| Training | Validation | Testing |
|----------|------------|---------|
| 70% | 15% | 15% |

## 🛠️ Tech Stack
Python 3.12, TensorFlow, Keras, PyTorch, OpenCV, NumPy, Pandas, Grad-CAM, Matplotlib, Seaborn, Streamlit.

## 🔄 Workflow
`Image Input → Preprocessing → Model Prediction → Grad-CAM Visualization → Dashboard`

## 🚀 Installation & Usage
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
pip install -r requirements.txt
streamlit run app.py
```

## 🔮 Future Scope
- Defect localization and severity estimation
- Ripeness assessment
- Automated fruit sorting
- Edge device deployment

## 👩‍💻 Author
**Isha** | B.Tech Computer Science & Engineering
