# 🏦 Direct Marketing Data Analysis

A comprehensive machine learning analysis of Portuguese banking institution's direct marketing campaigns to predict term deposit subscriptions.

## 📊 Project Overview

This project analyzes direct marketing campaign data from a Portuguese bank, where marketing efforts were conducted through phone calls to determine whether clients would subscribe to term deposits. The analysis implements multiple machine learning algorithms to predict subscription outcomes and compare their performance.

## 🎯 Objective

**Primary Goal**: Predict whether a client will subscribe to a bank term deposit (binary classification: 'yes' or 'no')

The classification target is based on the outcome of direct marketing campaigns that often required multiple contacts with the same client to assess their interest in the bank's term deposit product.

## 🔧 Technologies & Libraries

```python
# Core Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Scientific Computing
import scipy
from scipy.spatial.distance import pdist, squareform
```

## 📁 Project Structure

```
Direct-Marketing-Data-Analysis/
│
├── Marketing.py           # Main analysis script
├── README.md             # Project documentation
├── Training.csv          # Training dataset
├── Testing.csv           # Testing dataset
└── requirements.txt      # Dependencies (recommended)
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

### Running the Analysis

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Direct-Marketing-Data-Analysis.git
   cd Direct-Marketing-Data-Analysis
   ```

2. **Prepare your data**
   - Ensure `Training.csv` and `Testing.csv` are in the correct path
   - Update file paths in `Marketing.py` if necessary

3. **Execute the analysis**
   ```bash
   python Marketing.py
   ```

## 🔍 Methodology

### Data Preprocessing
- **Categorical Encoding**: One-hot encoding applied to categorical features (columns: 1,2,3,4,6,7,8,10,14,15)
- **Data Type Conversion**: All features converted to float for numerical processing
- **Feature Engineering**: Systematic handling of mixed data types

### Machine Learning Models

| Algorithm | Configuration | Purpose |
|-----------|---------------|---------|
| **Decision Tree** | `max_depth=6` | Handles categorical attributes effectively |
| **Random Forest** | `n_estimators=40, max_depth=10` | Ensemble method for improved accuracy |
| **Support Vector Machine** | Default SVC parameters | Non-linear classification capability |
| **Logistic Regression** | `C=5` | Linear baseline with regularization |

## 📈 Model Performance

The script outputs error rates for each algorithm:

```
DT Error  : [Decision Tree Error Rate]%
RF Error  : [Random Forest Error Rate]%  
SVM Error : [Support Vector Machine Error Rate]%
LR Error  : [Logistic Regression Error Rate]%
```

**Lower error rates indicate better model performance.**

## 📋 Data Features

The dataset contains various client attributes including:
- **Demographic Information**: Age, job, marital status, education
- **Financial History**: Credit default, housing loan, personal loan
- **Campaign Details**: Contact communication, previous campaigns
- **Economic Indicators**: Employment variation rate, consumer confidence

*Target Variable*: `y` - Term deposit subscription (yes/no)

## 🔬 Analysis Workflow

1. **Data Import**: Load training and testing datasets
2. **Preprocessing**: Encode categorical variables and convert data types
3. **Model Training**: Train four different classification algorithms
4. **Evaluation**: Calculate and compare error rates across models
5. **Results**: Output performance metrics for model comparison

## 📊 Key Insights

- **Decision Trees**: Excellent for interpretable results with categorical data
- **Random Forest**: Provides ensemble learning benefits and feature importance
- **SVM**: Effective for complex decision boundaries
- **Logistic Regression**: Offers probabilistic interpretations and baseline performance

## 🛠️ Future Enhancements

- [ ] Add cross-validation for robust performance estimation
- [ ] Implement feature importance analysis
- [ ] Include precision, recall, and F1-score metrics
- [ ] Add data visualization for exploratory analysis
- [ ] Hyperparameter tuning using GridSearchCV
- [ ] ROC curve and AUC analysis
- [ ] Feature selection techniques

## 📈 Model Comparison Framework

```python
# Suggested enhancement for comprehensive evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- **Dataset Source**: UCI Machine Learning Repository - Bank Marketing Dataset
- **Domain**: Banking and Financial Services
- **Application**: Direct Marketing Campaign Optimization

## 📧 Contact

For questions or collaboration opportunities, please reach out through GitHub issues or contact me at jahnaviisran12@gmail.com .

---

**⭐ Star this repository if you found it helpful!**