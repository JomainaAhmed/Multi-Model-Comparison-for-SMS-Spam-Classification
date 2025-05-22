# NLP Based SMS Spam Detection

## Project Overview

This project presents a comparative analysis of multiple machine learning algorithms applied to the task of SMS spam detection in the telecommunication sector. Using a real-world dataset from the UCI Machine Learning Repository, the study explores how well different classifiers distinguish between spam and ham (legitimate) messages.

## Algorithms Used

- Multinomial Naïve Bayes
- Gaussian Naïve Bayes
- Bernoulli Naïve Bayes
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

## Technologies and Tools

- Python  
- scikit-learn  
- NLTK  
- Seaborn & Matplotlib  
- Pandas & NumPy  
- Jupyter Notebook  

## Dataset

- Source: UCI Machine Learning Repository  
- Rows: 5572  
- Features: Message Label (`spam` or `ham`) and Message Content  

## Preprocessing Steps

- Data cleaning and deduplication  
- Label encoding (`ham` → `0`, `spam` → `1`)  
- Text normalization (lowercasing, punctuation removal, etc.)  
- Tokenization, stopword removal, stemming  
- Feature extraction using TF-IDF  

## Exploratory Data Analysis

- Visualization of spam/ham distribution
- Histograms showing message length trends
- Correlation heatmap
- Word clouds for spam and ham messages

## Model Evaluation

### Naïve Bayes Classifier Comparison

| Model                | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Gaussian NB          | 87.00%   | 51.00%    | 80.00% | 62.00%   |
| Multinomial NB       | 97.10%   | 100.00%   | 78.26% | 87.80%   |
| Bernoulli NB         | 98.00%   | 99.00%    | 88.00% | 93.00%   |

### Other Classifier Performance

| Model                | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Support Vector Classifier (SVC) | 97.58% | 97.48%    | 84.06% | 90.27%   |
| Logistic Regression  | 95.84%   | 97.03%    | 71.01% | 82.01%   |
| K-Nearest Neighbors (KNN) | 90.52% | 100.00%   | 28.99% | 44.94%   |

### After Hyperparameter Tuning

| Model                | Accuracy | Precision |
|---------------------|----------|-----------|
| SVC (Linear Kernel)  | 97.48%   | 97.10%    |
| Multinomial NB       | 97.10%   | 100.00%   |
| Logistic Regression  | 97.39%   | 97.43%    |
| KNN                  | 89.94%   | 100.00%   |

**Conclusion**: Multinomial Naïve Bayes provided the best results for SMS spam detection with an **accuracy of 97.10%** and **precision of 100%**, making it highly reliable especially for imbalanced datasets.

## Model Tuning

Hyperparameter tuning was performed using:
- Grid Search
- Random Search

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sms-spam-detection.git
   cd sms-spam-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Folder Structure

```
sms-spam-detection/
├── data/
├── notebooks/
├── src/
├── outputs/
├── requirements.txt
└── README.md
```

## Author

**Jomaina Hafiz Ahmed**  
School of Computer Science and Engineering  
Lovely Professional University, Phagwara, Punjab, India  
Email: jomaina.ahmed@gmail.com  

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

Thanks to the UCI Machine Learning Repository and all contributors whose tools and research made this study possible.

