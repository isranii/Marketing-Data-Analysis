\# 🏦 Direct Marketing Data Analysis



A comprehensive machine learning analysis of Portuguese banking institution's direct marketing campaigns to predict term deposit subscriptions.



\## 📊 Project Overview



This project analyzes direct marketing campaign data from a Portuguese bank, where marketing efforts were conducted through phone calls to determine whether clients would subscribe to term deposits. The analysis implements multiple machine learning algorithms to predict subscription outcomes and compare their performance.



\## 🎯 Objective



\*\*Primary Goal\*\*: Predict whether a client will subscribe to a bank term deposit (binary classification: 'yes' or 'no')



The classification target is based on the outcome of direct marketing campaigns that often required multiple contacts with the same client to assess their interest in the bank's term deposit product.



\## 🔧 Technologies \& Libraries



```python

\# Core Libraries

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



\# Machine Learning

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.linear\_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier



\# Scientific Computing

import scipy

from scipy.spatial.distance import pdist, squareform

```



\## 📁 Project Structure



```

Direct-Marketing-Data-Analysis/

│

├── Marketing.py           # Main analysis script

├── README.md             # Project documentation

├── Training.csv          # Training dataset

├── Testing.csv           # Testing dataset

└── requirements.txt      # Dependencies (recommended)

```



\## 🚀 Getting Started



\### Prerequisites



```bash

pip install numpy pandas matplotlib scikit-learn scipy

```



\### Running the Analysis



1\. \*\*Clone the repository\*\*

&nbsp;  ```bash

&nbsp;  git clone https://github.com/yourusername/Direct-Marketing-Data-Analysis.git

&nbsp;  cd Direct-Marketing-Data-Analysis

&nbsp;  ```



2\. \*\*Prepare your data\*\*

&nbsp;  - Ensure `Training.csv` and `Testing.csv` are in the correct path

&nbsp;  - Update file paths in `Marketing.py` if necessary



3\. \*\*Execute the analysis\*\*

&nbsp;  ```bash

&nbsp;  python Marketing.py

&nbsp;  ```



\## 🔍 Methodology



\### Data Preprocessing

\- \*\*Categorical Encoding\*\*: One-hot encoding applied to categorical features (columns: 1,2,3,4,6,7,8,10,14,15)

\- \*\*Data Type Conversion\*\*: All features converted to float for numerical processing

\- \*\*Feature Engineering\*\*: Systematic handling of mixed data types



\### Machine Learning Models



| Algorithm | Configuration | Purpose |

|-----------|---------------|---------|

| \*\*Decision Tree\*\* | `max\_depth=6` | Handles categorical attributes effectively |

| \*\*Random Forest\*\* | `n\_estimators=40, max\_depth=10` | Ensemble method for improved accuracy |

| \*\*Support Vector Machine\*\* | Default SVC parameters | Non-linear classification capability |

| \*\*Logistic Regression\*\* | `C=5` | Linear baseline with regularization |



\## 📈 Model Performance



The script outputs error rates for each algorithm:



```

DT Error  : \[Decision Tree Error Rate]%

RF Error  : \[Random Forest Error Rate]%  

SVM Error : \[Support Vector Machine Error Rate]%

LR Error  : \[Logistic Regression Error Rate]%

```



\*\*Lower error rates indicate better model performance.\*\*



\## 📋 Data Features



The dataset contains various client attributes including:

\- \*\*Demographic Information\*\*: Age, job, marital status, education

\- \*\*Financial History\*\*: Credit default, housing loan, personal loan

\- \*\*Campaign Details\*\*: Contact communication, previous campaigns

\- \*\*Economic Indicators\*\*: Employment variation rate, consumer confidence



\*Target Variable\*: `y` - Term deposit subscription (yes/no)



\## 🔬 Analysis Workflow



1\. \*\*Data Import\*\*: Load training and testing datasets

2\. \*\*Preprocessing\*\*: Encode categorical variables and convert data types

3\. \*\*Model Training\*\*: Train four different classification algorithms

4\. \*\*Evaluation\*\*: Calculate and compare error rates across models

5\. \*\*Results\*\*: Output performance metrics for model comparison



\## 📊 Key Insights



\- \*\*Decision Trees\*\*: Excellent for interpretable results with categorical data

\- \*\*Random Forest\*\*: Provides ensemble learning benefits and feature importance

\- \*\*SVM\*\*: Effective for complex decision boundaries

\- \*\*Logistic Regression\*\*: Offers probabilistic interpretations and baseline performance



\## 🛠️ Future Enhancements



\- \[ ] Add cross-validation for robust performance estimation

\- \[ ] Implement feature importance analysis

\- \[ ] Include precision, recall, and F1-score metrics

\- \[ ] Add data visualization for exploratory analysis

\- \[ ] Hyperparameter tuning using GridSearchCV

\- \[ ] ROC curve and AUC analysis

\- \[ ] Feature selection techniques



\## 📈 Model Comparison Framework



```python

\# Suggested enhancement for comprehensive evaluation

from sklearn.metrics import classification\_report, confusion\_matrix, roc\_auc\_score



def evaluate\_model(y\_true, y\_pred, model\_name):

&nbsp;   accuracy = accuracy\_score(y\_true, y\_pred)

&nbsp;   report = classification\_report(y\_true, y\_pred)

&nbsp;   cm = confusion\_matrix(y\_true, y\_pred)

&nbsp;   

&nbsp;   print(f"\\n{model\_name} Performance:")

&nbsp;   print(f"Accuracy: {accuracy:.4f}")

&nbsp;   print("Classification Report:")

&nbsp;   print(report)

```



\## 🤝 Contributing



1\. Fork the repository

2\. Create your feature branch (`git checkout -b feature/AmazingFeature`)

3\. Commit your changes (`git commit -m 'Add some AmazingFeature'`)

4\. Push to the branch (`git push origin feature/AmazingFeature`)

5\. Open a Pull Request



\## 📝 License



This project is licensed under the MIT License - see the \[LICENSE](LICENSE) file for details.



\## 🔗 References



\- \*\*Dataset Source\*\*: UCI Machine Learning Repository - Bank Marketing Dataset

\- \*\*Domain\*\*: Banking and Financial Services

\- \*\*Application\*\*: Direct Marketing Campaign Optimization



\## 📧 Contact



For questions or collaboration opportunities, please reach out through GitHub issues or contact me at jahnaviisrani12@gmail.com .



---



\*\*⭐ Star this repository if you found it helpful!\*\*

