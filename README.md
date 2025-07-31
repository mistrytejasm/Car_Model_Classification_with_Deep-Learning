# README.md

## 🚗 Car Model Classification with out-of-distribution (OOD) Detection using Multi-Head Architecture

### 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Challenges with Standard Classification](#-challenges-with-standard-classification)
- [Our OOD Solution](#-our-ood-solution)
- [Technical Implementation](#-technical-implementation)
- [Results \& Performance](#-results--performance)
- [Installation \& Usage](#-installation--usage)
- [Model Architecture](#-model-architecture)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)


### 🎯 Problem Statement

This project addresses the challenge of **robust car model classification** that can distinguish between **4 specific Indian car models** while simultaneously detecting **out-of-distribution (OOD) images** that don't belong to any car category.

**Target Classes:**

- Maruti Suzuki Baleno
- Maruti Suzuki Brezza
- Maruti Suzuki Swift
- Maruti Suzuki WagonR


### 📊 Dataset

The dataset includes:

- **Training/Validation/Test Sets**: Images of 4 car models organized in labeled folders
- **OOD Detection**: Additional "Not_Car" folder containing random images (animals, objects, people, etc.)
- **Data Augmentation**: Applied to improve model generalization and robustness

```
dataset/
├── train/
│   ├── Maruti_Suzuki_Baleno/
│   ├── Maruti_Suzuki_Brezza/
│   ├── Maruti_Suzuki_Swift/
│   ├── Maruti_Suzuki_WagonR/
│   └── Not_Car/              # OOD samples
├── val/
└── test/
```


### ⚠️ Challenges with Standard Classification

#### **The Core Problem**

Standard image classification models are trained to classify input images into one of the predefined classes. However, this approach has a critical flaw:

**When a random image (not a car) is input to a standard car classifier:**

- ⚠️ The model **forcefully assigns** it to one of the 4 car classes
- ⚠️ **High confidence scores** are often given to completely wrong predictions
- ⚠️ **No mechanism exists** to detect that the input doesn't belong to any car category
- ⚠️ **False positives** occur when classifying random objects as specific car models


#### **Real-World Examples of the Problem:**

| Input Image | Standard Model Prediction | Issue |
| :-- | :-- | :-- |
| 🐕 Dog | "Swift with 89% confidence" | Completely wrong classification |
| 🏠 House | "Baleno with 76% confidence" | Non-car classified as car |
| 👤 Person | "WagonR with 82% confidence" | High confidence on irrelevant input |
| 🌳 Tree | "Brezza with 91% confidence" | Model has no "rejection" capability |

### 💡 Our OOD Solution

#### **What is Out-of-Distribution (OOD) Detection?**

OOD detection is a machine learning technique that enables models to:

1. **Identify** when input data doesn't belong to any of the training categories
2. **Reject** or flag suspicious inputs instead of forcing classification
3. **Maintain reliability** by avoiding confident wrong predictions on irrelevant data

#### **How Our Solution Works:**

Our **Multi-Head Architecture** approach uses:

1. **🎯 Car Classification Head**: Classifies among 4 car models (when input is actually a car)
2. **🚨 OOD Detection Head**: Determines if the input is a car vs. not-a-car (binary classification)
```
Input Image → EfficientNet-B0 Backbone → Shared Features
                                              ↓
                                    ┌─── Car Classifier Head (4 classes)
                                    └─── OOD Detector Head (binary: car/not-car)
```


#### **Two-Stage Decision Process:**

**Stage 1: OOD Detection**

- 🔍 Is this image a car or not?
- If OOD score < 0.5 → "Not a Car" (reject for car classification)
- If OOD score ≥ 0.5 → Proceed to car classification

**Stage 2: Car Classification** (only if Stage 1 confirms it's a car)

- 🚗 Which of the 4 car models is this?
- Return specific car model with confidence score


### 🛠️ Technical Implementation

#### **Model Architecture**

- **Backbone**: EfficientNet-B0 (pre-trained on ImageNet)
- **Transfer Learning**: Frozen feature layers + fine-tuned classifier heads
- **Multi-Head Design**: Shared features, separate classification heads
- **Input Size**: 224×224 RGB images
- **Optimization**: Adam optimizer with learning rate scheduling and early stopping


#### **Loss Function**

**Dual Loss Approach:**

```python
total_loss = car_weight * car_classification_loss + ood_weight * ood_detection_loss
```

- **Car Classification Loss**: CrossEntropyLoss (only for actual car images)
- **OOD Detection Loss**: Binary CrossEntropyLoss (for all images)
- **Weight Balance**: Optimized for both tasks simultaneously


#### **Training Strategy**

- **Data Augmentation**: Random flips, rotations, color jittering, resized crops
- **Early Stopping**: Prevents overfitting with patience-based monitoring
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
- **Dual Metrics Tracking**: Separate accuracies for car classification and OOD detection


### 📈 Results \& Performance

Our multi-head OOD approach significantly outperforms standard classification:

<img width="1156" height="623" alt="image" src="https://github.com/user-attachments/assets/7404ad99-4fd5-40f7-bd77-aea246cd293f" />


#### **Standard vs OOD Comparison:**

| Metric | Standard Classification | Our OOD Solution |
| :-- | :-- | :-- |
| **Car Classification Accuracy** | ~76% | **~85%** ✅ |
| **False Positive Rate** | Very High ⚠️ | **Low** ✅ |
| **OOD Detection Accuracy** | N/A | **~92%** ✅ |
| **Robustness** | Poor ⚠️ | **Excellent** ✅ |

#### **Key Improvements:**

- ✅ **92% accuracy** in detecting non-car images
- ✅ **85% accuracy** in classifying actual car models
- ✅ **Robust rejection** of irrelevant inputs
- ✅ **No more false car classifications** on random images
- ✅ **Production-ready reliability** for real-world deployment


### 🚀 Installation \& Usage

#### **Prerequisites**

```bash
pip install torch torchvision matplotlib tqdm pillow pathlib
```


#### **Quick Start**

1. **Clone the Repository**
```bash
https://github.com/mistrytejasm/Car_Model_Classification_with_Deep-Learning.git
cd car-classification-ood
```

2. **Prepare Dataset**
```bash
# Organize your data in the required structure
dataset/
├── train/
├── val/
└── test/
```

### 📊 Model Files \& Metrics

**Saved Model Components:**

- `car_classifier_TIMESTAMP.pth` - Model weights (~22 MB)
- `car_classifier_TIMESTAMP_full.pth` - Complete model (~25 MB)
- `car_classifier_TIMESTAMP_results.pkl` - Training metrics
- `car_classifier_TIMESTAMP_metadata.json` - Model configuration

**Training Metrics Tracked:**

- Total Loss, Car Classification Loss, OOD Detection Loss
- Car Classification Accuracy, OOD Detection Accuracy
- Learning rate progression, early stopping events


### 🔮 Future Improvements

- [ ] **Uncertainty Quantification**: Add model confidence estimation
- [ ] **More Car Models**: Extend to additional Indian car manufacturers
- [ ] **Real-time Inference**: Optimize for mobile and edge deployment

### 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments

- **Transfer Learning**: EfficientNet-B0 pre-trained on ImageNet
- **Framework**: PyTorch ecosystem
- **Inspiration**: Modern OOD detection research and practical deployment needs


### 📞 Contact

- **Author**: TejasH Mistry
- **Email**: mistrytejasm@gmail.com
- **LinkedIn**: [[Your LinkedIn Profile]](https://www.linkedin.com/in/tejash-mistry-7789b9184/)

**🔧 Built with passion for robust AI systems that work reliably in real-world scenarios!**

[^1]: https://colab.research.google.com/drive/1E7IoFU2jr_lfJfR1nAH0c8N8QHFuOAlf

