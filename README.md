# ModelStack Benchmark Studio 🔬

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.50.0-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.7.2-orange.svg)](https://scikit-learn.org/)

> **An Interactive Machine Learning Benchmark Platform for Ensemble Model Evaluation and Comparison**

ModelStack Benchmark Studio is a comprehensive web-based tool designed to demonstrate the effectiveness of Bagging-based ensemble methods in improving model robustness, accuracy, and generalization. Built with Streamlit, this platform provides an intuitive interface for training, comparing, and visualizing machine learning models with a focus on educational accessibility and practical applicability.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Research Background](#research-background)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Supported Models](#supported-models)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Results & Findings](#results--findings)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

Machine learning practitioners often face challenges with model variance, overfitting, and poor generalization—particularly when working with small, noisy, or imbalanced datasets. **ModelStack Benchmark Studio** addresses these issues by implementing Bootstrap Aggregating (Bagging) with diverse base learners to create robust ensemble models.

This tool is designed for:
- **Students** learning about ensemble methods and model evaluation
- **Researchers** conducting comparative studies on ML algorithms
- **Practitioners** seeking reliable model selection for real-world applications
- **Educators** teaching machine learning concepts interactively

---

## ✨ Key Features

### 🤖 **Comprehensive Model Support**
- Multiple base learners: Decision Trees, Logistic Regression, K-Nearest Neighbors, SVM, Random Forest, and more
- Bagging ensemble implementation for variance reduction
- Side-by-side comparison of individual vs. ensemble performance

### 📊 **Interactive Data Management**
- Built-in benchmark datasets (Diabetes, Iris, Digits)
- Custom dataset upload support (CSV format)
- Automated data preprocessing and train-test splitting
- Real-time data visualization and statistics

### 📈 **Advanced Visualization**
- Performance metrics comparison charts
- Confusion matrices and classification reports
- Feature importance analysis
- ROC curves and precision-recall curves
- Interactive Plotly-based visualizations

### 🔍 **Model Explainability**
- Feature importance rankings
- SHAP value integration (optional)
- Decision boundary visualization
- Model behavior interpretation tools

### 💾 **Experiment Tracking**
- SQLite-based experiment database
- Historical performance tracking
- Experiment comparison across time
- Export results to CSV/JSON

### 🎨 **User-Friendly Interface**
- Clean, intuitive Streamlit design
- Responsive layout for various screen sizes
- Step-by-step workflow guidance
- Real-time feedback and progress indicators

---

## 🔬 Research Background

### Problem Statement

Individual machine learning models frequently exhibit:
- **High Variance**: Sensitivity to training data fluctuations
- **Overfitting**: Excellent training performance but poor generalization
- **Instability**: Inconsistent predictions across different data samples

These issues are amplified in domains like:
- 🏥 Healthcare (limited patient data)
- 💰 Finance (market volatility and noise)
- 🔒 Cybersecurity (imbalanced threat datasets)

### Solution: Bagging Ensemble Method

**Bootstrap Aggregating (Bagging)** addresses these challenges by:
1. Creating multiple bootstrap samples from the training data
2. Training diverse base models on each sample
3. Aggregating predictions through voting (classification) or averaging (regression)

**Benefits:**
- ✅ Reduced variance without increasing bias
- ✅ Improved generalization to unseen data
- ✅ Enhanced model stability and robustness
- ✅ Lower risk of overfitting

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/navneetshukla/modelstack-benchmark-studio.git
   cd modelstack-benchmark-studio
   ```

2. **Create Virtual Environment** (Optional but Recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   streamlit --version
   python -c "import sklearn; print(sklearn.__version__)"
   ```

---

## ⚡ Quick Start

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### First-Time Setup

1. **Navigate to Data Management** 📊
   - Select a built-in dataset (Iris, Diabetes, or Digits)
   - Or upload your own CSV file

2. **Configure Model Training** 🤖
   - Choose base learners (e.g., Decision Tree, Logistic Regression)
   - Select Bagging ensemble option
   - Adjust hyperparameters as needed

3. **Train and Compare** 📈
   - Click "Train Models"
   - View real-time performance metrics
   - Compare individual vs. ensemble results

4. **Analyze Results** 🔍
   - Explore confusion matrices
   - Review feature importance
   - Export experiment results

---

## 📖 Usage Guide

### 1. Data Management Page

**Upload Custom Data:**
```python
# Your CSV should have:
# - Features in columns (numerical or categorical)
# - Target variable in the last column
# Example: iris.csv with 4 features + 1 target
```

**Data Preprocessing:**
- Automatic handling of missing values
- Label encoding for categorical variables
- Feature scaling (optional)
- Train-test split configuration (default: 80-20)

### 2. Model Training Page

**Selecting Models:**
- Individual classifiers: Decision Tree, Logistic Regression, KNN, SVM, etc.
- Ensemble: Bagging with configurable base estimators

**Hyperparameter Tuning:**
```python
# Example configurations
Bagging Parameters:
- n_estimators: 10-100
- max_samples: 0.5-1.0
- max_features: 0.5-1.0

Decision Tree:
- max_depth: 3-20
- min_samples_split: 2-10
```

### 3. Model Comparison Page

Compare metrics including:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Score
- Training Time

### 4. Model Explainability Page

Understand model decisions through:
- Feature importance rankings
- SHAP value plots (if installed)
- Decision boundary visualizations
- Prediction confidence analysis

---

## 📁 Project Structure

```
modelstack-benchmark-studio/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
├── README.md                  # This file
│
├── pages/                     # Streamlit multi-page app
│   ├── 1_📊_Data_Management.py
│   ├── 2_🤖_Model_Training.py
│   ├── 3_📈_Model_Comparison.py
│   └── 4_🔍_Model_Explainability.py
│
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── database.py           # SQLite database operations
│   ├── preprocessing.py      # Data preprocessing functions
│   ├── models.py             # Model training and evaluation
│   ├── visualization.py      # Plotting and charting functions
│   └── metrics.py            # Performance metrics calculation
│
├── data/                      # Dataset storage
│   ├── benchmark/            # Built-in benchmark datasets
│   │   ├── iris.csv
│   │   ├── diabetes.csv
│   │   └── digits.csv
│   └── uploads/              # User-uploaded datasets
│
├── experiments/               # Saved experiments
│   ├── experiments.db        # SQLite database
│   └── models/               # Serialized trained models
│
├── docs/                      # Documentation
│   ├── API.md
│   ├── METHODOLOGY.md
│   └── RESEARCH_PAPER.pdf
│
└── tests/                     # Unit tests
    ├── test_preprocessing.py
    ├── test_models.py
    └── test_visualization.py
```

---

## 🤖 Supported Models

### Base Learners

| Model | Type | Best For |
|-------|------|----------|
| Decision Tree | Classification/Regression | Interpretable, non-linear patterns |
| Logistic Regression | Classification | Linear separable data, baseline |
| K-Nearest Neighbors | Classification/Regression | Local pattern recognition |
| Support Vector Machine | Classification | High-dimensional data |
| Random Forest | Classification/Regression | General-purpose, robust |
| Naive Bayes | Classification | Text classification, fast training |
| Gradient Boosting | Classification/Regression | Complex patterns, high accuracy |

### Ensemble Methods

| Method | Description | Advantage |
|--------|-------------|-----------|
| Bagging | Bootstrap + Aggregating | Variance reduction |
| Voting Classifier | Majority voting | Combines diverse models |
| Stacking | Meta-learner on predictions | Captures model strengths |

---

## 📊 Datasets

### Built-in Benchmark Datasets

1. **Iris Dataset** 🌸
   - **Samples:** 150
   - **Features:** 4 (sepal/petal measurements)
   - **Classes:** 3 (setosa, versicolor, virginica)
   - **Use Case:** Multi-class classification

2. **Diabetes Dataset** 🏥
   - **Samples:** 768
   - **Features:** 8 (medical measurements)
   - **Classes:** 2 (diabetic, non-diabetic)
   - **Use Case:** Binary classification, healthcare

3. **Digits Dataset** ✍️
   - **Samples:** 1,797
   - **Features:** 64 (8x8 pixel images)
   - **Classes:** 10 (digits 0-9)
   - **Use Case:** Image classification, handwriting recognition

### Custom Dataset Requirements

- **Format:** CSV file
- **Structure:** Features in columns, target in last column
- **Encoding:** UTF-8
- **Missing Values:** Handled automatically
- **Size Limit:** 50 MB (configurable)

---

## 🔬 Methodology

### Bagging Implementation

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Create bagging ensemble
bagging_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=50,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    random_state=42
)

# Train model
bagging_model.fit(X_train, y_train)

# Predict
y_pred = bagging_model.predict(X_test)
```

### Evaluation Metrics

**Classification:**
- Accuracy: Overall correctness
- Precision: Positive prediction accuracy
- Recall: Positive case detection rate
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC: Area under ROC curve

**Regression:**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

---

## 📈 Results & Findings

### Key Research Outcomes

Based on experiments conducted on benchmark datasets from scikit-learn:

| Dataset | Individual Best | Bagging Ensemble | Improvement |
|---------|----------------|------------------|-------------|
| Iris (KNN) | 95.3% | 97.1% | +1.8% |
| Diabetes (DT) | 73.2% | 80.5% | +7.3% |
| Digits (LR) | 93.5% | 96.2% | +2.7% |

### Variance Reduction

Bagging demonstrated significant variance reduction across all datasets:
- **Iris:** 50% reduction (0.008 → 0.004)
- **Diabetes:** 57% reduction (0.021 → 0.009)
- **Digits:** 53% reduction (0.015 → 0.007)

### Robustness Under Noise and Imbalance

**Noise Resilience (10% artificial noise):**
- Diabetes Base DT: 73.2% → 66.4% (dropped 6.8%)
- Diabetes Bagging: 80.5% → 78.9% (dropped only 1.6%)

**Imbalance Tolerance:**
- Digits Logistic Regression: 93.5% → 89.1%
- Digits Bagging: 96.2% → 94.7%

### Educational Impact

Survey results from MCA students (n=30):
- **90%** found Bagging easier to interpret than Boosting
- **87%** reported improved understanding via Streamlit visualization
- **93%** agreed that Bagging enhanced robustness perception

### Performance Metrics Summary

| Dataset | Model | Accuracy | Precision | Recall | F1-Score | Variance |
|---------|-------|----------|-----------|--------|----------|----------|
| Diabetes | Decision Tree | 73.2% | 0.72 | 0.71 | 0.71 | 0.021 |
| Diabetes | Bagging (DT) | 80.5% | 0.81 | 0.80 | 0.80 | 0.009 |
| Iris | KNN | 95.3% | 0.95 | 0.95 | 0.95 | 0.008 |
| Iris | Bagging (KNN) | 97.1% | 0.97 | 0.97 | 0.97 | 0.004 |
| Digits | Logistic Reg. | 93.5% | 0.93 | 0.93 | 0.93 | 0.015 |
| Digits | Bagging (LR) | 96.2% | 0.96 | 0.96 | 0.96 | 0.007 |

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

1. **Bug Reports**: Open an issue with detailed reproduction steps
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit pull requests with enhancements
4. **Documentation**: Improve docs, add examples, fix typos
5. **Testing**: Write unit tests, report edge cases

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/navneetshukla/modelstack-benchmark-studio.git

# Create a feature branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Submit pull request
```

### Code Style Guidelines

- Follow PEP 8 conventions
- Add docstrings to functions
- Include type hints where appropriate
- Write unit tests for new features

---

## 📚 Citation

If you use this tool in your research or projects, please cite:

```bibtex
@inproceedings{shukla2022enhancing,
  title={Enhancing Model Robustness through Bagging-Based Ensemble Learning},
  author={Shukla, Navneet},
  month={March},
  year={2022},
  supervisor={Mishra, Padma},
  institution={Thakur Institute of Management Studies, Career Development \& Research (TIMSCDR)},
  note={Interactive Streamlit-based platform for ensemble learning visualization}
}
```

**Research Paper:**
```
Shukla, N., Siddiquie, A., & Mishra, P. (2022). 
"Enhancing Model Robustness through Bagging-Based Ensemble Learning." 
tinyML Research Symposium'22, March 2022, San Jose, CA.
Thakur Institute of Management Studies, Career Development & Research (TIMSCDR).
```

**Authors:**
- **Navneet Shukla** (182) - shuklanavneet2817@gmail.com
- **Aadil Siddiquie** (183) - aadil.siddiquie183@gmail.com
- **Research Supervisor:** Dr. Padma Mishra

**Institution:**
Thakur Institute of Management Studies, Career Development & Research (TIMSCDR)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2022 Navneet Shukla, Aadil Siddiquie
Thakur Institute of Management Studies, Career Development & Research (TIMSCDR)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📞 Contact

**Authors:**
- **Navneet Shukla** (182)
  - Email: shuklanavneet2817@gmail.com
  - Institution: Thakur Institute of Management Studies, Career Development & Research (TIMSCDR)

- **Aadil Siddiquie** (183)
  - Email: aadil.siddiquie183@gmail.com
  - Institution: Thakur Institute of Management Studies, Career Development & Research (TIMSCDR)

**Research Supervisor:**
- **Dr. Padma Mishra**
  - Institution: Thakur Institute of Management Studies, Career Development & Research (TIMSCDR)

### Support

- **Issues:** [GitHub Issues](https://github.com/navneetshukla/modelstack-benchmark-studio/issues)
- **Discussions:** [GitHub Discussions](https://github.com/navneetshukla/modelstack-benchmark-studio/discussions)
- **Documentation:** [Wiki](https://github.com/navneetshukla/modelstack-benchmark-studio/wiki)
- **Research Paper:** Available at tinyML Research Symposium'22 Proceedings

---

## 🙏 Acknowledgments

- **Dr. Padma Mishra** - Research Supervisor and Mentor
- **Thakur Institute of Management Studies, Career Development & Research (TIMSCDR)** - Institutional Support
- **tinyML Research Symposium'22** - Conference Platform
- **scikit-learn** team for excellent ML library
- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations
- Open-source community contributors
- MCA students who participated in the usability study (n=30)

---

## 🗺️ Roadmap

### Version 1.0 (Current)
- ✅ Core bagging implementation
- ✅ Benchmark datasets integration
- ✅ Interactive web interface
- ✅ Basic model comparison

### Version 1.1 (Planned)
- 🔲 Additional ensemble methods (AdaBoost, XGBoost)
- 🔲 Automated hyperparameter tuning
- 🔲 Advanced SHAP integration
- 🔲 Multi-language support

### Version 2.0 (Future)
- 🔲 Deep learning model support
- 🔲 Cloud deployment options
- 🔲 Real-time model monitoring
- 🔲 Collaborative experiment sharing

---

## 📊 Statistics

![GitHub stars](https://img.shields.io/github/stars/navneetshukla/modelstack-benchmark-studio?style=social)
![GitHub forks](https://img.shields.io/github/forks/navneetshukla/modelstack-benchmark-studio?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/navneetshukla/modelstack-benchmark-studio?style=social)

---

<div align="center">

**Built with ❤️ for the Machine Learning Community**

[⬆ Back to Top](#modelstack-benchmark-studio-)

</div>
