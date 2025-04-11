# SC4001 Project - Clothing Classification  
*Deep Learning Course Project | Nanyang Technological University (NTU)*  

## Table of Contents  
- [SC4001 Project - Clothing Classification](#sc4001-project---clothing-classification)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
    - [📌 Introduction](#-introduction)
    - [🎯 Key Features](#-key-features)
  - [Team Members](#team-members)
  - [Technical Implementation](#technical-implementation)
    - [🛠️ Dependencies](#️-dependencies)
    - [🏗️ Code Structure](#️-code-structure)
    - [⚙️ Key Techniques Explained](#️-key-techniques-explained)
  - [Getting Started](#getting-started)
    - [📥 Installation](#-installation)
    - [📚 Documentation](#-documentation)
  - [Results \& Findings](#results--findings)
  - [References](#references)

---

## Project Overview  

### 📌 Introduction  
This project explores the performance of **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** for clothing classification on the Fashion-MNIST dataset. We investigate how architectural biases and parameter efficiency impact model generalization, particularly in limited-data regimes. Our work highlights CNN's superiority on small-scale datasets and introduces advanced techniques like **Decoupled MixUp Regularization** and **Adaptive Deformable Convolutions** to further boost performance.  

### 🎯 Key Features  
- **Comparative Analysis**: CNN vs. ViT on Fashion-MNIST.  
- **Advanced Techniques**:  
  - *Decoupled MixUp Regularization* for input-space augmentation.  
  - *Adaptive Deformable Convolutions* for dynamic spatial feature learning.  
- **Parameter-Efficient Adaptation**: Vision Prompt Tuning (VPT) for ViTs.  
- **Synergistic Optimization**: Combining MixUp with deformable convolutions achieves **92.79% test accuracy**.  

---

## Team Members  
*College of Computing & Data Science, NTU*  
| Name                | GitHub Account                                |
| ------------------- | --------------------------------------------- |
| Allan Rooney Nounke | [TheNou1](https://github.com/TheNou1)         |
| Liangrui Zhang      | [dreamer-zlr](https://github.com/dreamer-zlr) |
| Ruilizhen Hu        | [HuRuilizhen](https://github.com/HuRuilizhen) |

---

## Technical Implementation  

### 🛠️ Dependencies  
- **Python 3.13**  
- **PyTorch 2.6.0** (with `torch.mps` backend for Apple Silicon GPU acceleration if available)  
- **TorchVision 0.21.0**  
- **NumPy**, **Matplotlib**  

### 🏗️ Code Structure  
```  
SC4001-Project/  
├── models/                     # Model architectures (CNN, ViT, DCNN)  
├── utils/                      # Data loaders, augmentation, and helper functions  
├── docs/                       # Project documentation and technical reports  
└── main_<model_name>.ipynb     # Entry script for training and evaluation  
```  

### ⚙️ Key Techniques Explained  
1. **Vision Transformer (ViT)**  
   - *Patch Embeddings*: Split images into 4×4 patches for linear projection.  
   - *Position Embeddings*: Learn spatial relationships between patches.  
   - *Visual Prompt Tuning (VPT)*: Adapt pre-trained ViTs with task-specific prompts.  

2. **CNN with Advanced Innovations**  
   - *Deformable Convolutions*: Learn dynamic receptive fields for geometric invariance.  
   - *MixUp Regularization*: Generate interpolated training samples to smooth decision boundaries.  

---

## Getting Started  

### 📥 Installation  
```bash  
git clone https://github.com/HuRuilizhen/SC4001-Project.git  
cd SC4001-Project  
pip install -r requirements.txt  
```  

### 📚 Documentation  
- Full technical details: See `docs/SC4001_Project.pdf`.  
- Training configuration: Refer to `configs/train_config.yaml`.  

---

## Results & Findings  
| Model            | Test Accuracy (%) | Test Loss  |
| ---------------- | ----------------- | ---------- |
| Baseline CNN     | 92.18             | 0.2541     |
| CNN + MixUp      | 91.96             | 0.2381     |
| Deformable CNN   | 92.16             | 0.2469     |
| **DCNN + MixUp** | **92.79**         | **0.2133** |

**Key Insights**:  
- MixUp reduces overconfidence in predictions despite slight accuracy trade-offs.  
- Deformable convolutions improve geometric invariance for complex classes (e.g., dresses vs. coats).  
- Hybrid optimization (DCNN + MixUp) achieves **16% lower test loss** than baseline CNN.  

---

## References  
1. Alexey Dosovitskiy et al. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).  
2. Zhu et al. [Deformable ConvNets V2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168).  
3. Han Xiao et al. [Fashion-MNIST: A Novel Image Dataset](https://arxiv.org/abs/1708.07747).  
4. Full citation list: See `docs/project_report.pdf`.  

---