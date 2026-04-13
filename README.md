# American Sign Language (ASL) Classification

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Google Colab](https://img.shields.io/badge/Google_Colab-Environment-yellow.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)
![NumPy](https://img.shields.io/badge/NumPy-Data_Manipulation-blueviolet.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-success.svg)
![Matplotlib/Seaborn](https://img.shields.io/badge/Matplotlib_&_Seaborn-Data_Visualization-lightgrey.svg)

This project aims to classify images of the American Sign Language (ASL) alphabet using a variety of machine learning techniques. The pipeline extracts critical visual features from images and trains several robust CPU-based models entirely within a Google Colab environment, leveraging Google Drive for data storage and output persistence.

## 📝 Dataset
- **Data Source:** [ASL Alphabet on Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet)
- The dataset comprises 29 classes, representing the 26 letters of the English alphabet (A-Z) along with 'space', 'del', and 'nothing'.

## 🛠️ Technologies & Libraries Used
- **Environment:** Google Colab (CPU mode)
- **Storage:** Google Drive (Mounted for input data & output export)
- **Language:** Python
- **Libraries:**
  - **Computer Vision:** `OpenCV` (cv2), `scikit-image`
  - **Machine Learning:** `scikit-learn`
  - **Data Manipulation:** `NumPy`, `Pandas`
  - **Visualization:** `Matplotlib`, `Seaborn`
  - **Utilities:** `tqdm`, `joblib`, `gc`

## 🧠 Approach & Architecture

### 1. Feature Extraction
To maintain low computational overhead and allow execution on CPU, we combine several powerful visual descriptors instead of using Deep Learning (CNNs):
- **Global Features:**
  - **HOG (Histogram of Oriented Gradients):** Captures edge directions and shape structures.
  - **Color Histogram:** Describes the color distribution in HSV space.
  - **Hu Moments:** Extracts structural features invariant to rotation and scale.
- **Local Features (Bag of Visual Words):**
  - **SIFT & ORB:** Local point descriptors extracted and clustered via `MiniBatchKMeans` (Vocabulary Size = 50) to create a visual word frequency histogram.

### 2. Preprocessing & ML Pipeline
- **Scaling:** Features are standardized using `StandardScaler`.
- **Dimensionality Reduction:** `PCA` (Principal Component Analysis) is applied to keep 95% of the variance, enhancing training speed and reducing noise.
- **Data Splits:** 80% Train / 10% Validation / 10% Test Split.
- **Models Evaluated:** Random Forest, Logistic Regression, SVM (Linear & RBF Kernels), AdaBoost, and Bagging Classifiers.

---

## 📊 Metrics and Results

### Validation Set (10% Split)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest** | 99.37% | 99.38% | 99.37% | 99.37% |
| **Log Regression** | 99.55% | 99.55% | 99.55% | 99.55% |
| **SVM Linear** | 99.76% | 99.76% | 99.76% | 99.76% |
| **SVM RBF** | 99.74% | 99.74% | 99.74% | 99.74% |
| **AdaBoost** | 54.70% | 58.48% | 54.70% | 55.55% |
| **Bagging** | 95.14% | 95.29% | 95.14% | 95.15% |

<img src="./Output/Charts/Validation/validation_bar_chart.png" alt="Validation Metrics Bar Chart" width="600" />

#### Validation Confusion Matrices
<details>
<summary>Click to view Validation Confusion Matrices</summary>

- <img src="./Output/Charts/Validation/validation_SVM_RBF_cm.png" alt="SVM RBF" width="400" />
- <img src="./Output/Charts/Validation/validation_SVM_Linear_cm.png" alt="SVM Linear" width="400" />
- <img src="./Output/Charts/Validation/validation_Log_Regression_cm.png" alt="Logistic Regression" width="400" />
- <img src="./Output/Charts/Validation/validation_Random_Forest_cm.png" alt="Random Forest" width="400" />
- <img src="./Output/Charts/Validation/validation_Bagging_cm.png" alt="Bagging" width="400" />
- <img src="./Output/Charts/Validation/validation_AdaBoost_cm.png" alt="AdaBoost" width="400" />
</details>

---

### Test Split (10% Unseen during standard train)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest** | 99.06% | 99.07% | 99.06% | 99.06% |
| **Log Regression** | 99.37% | 99.37% | 99.37% | 99.37% |
| **SVM Linear** | 99.76% | 99.76% | 99.76% | 99.76% |
| **SVM RBF** | 99.82% | 99.82% | 99.82% | 99.82% |
| **AdaBoost** | 56.28% | 59.84% | 56.28% | 57.19% |
| **Bagging** | 94.54% | 94.78% | 94.54% | 94.58% |


<img src="./Output/Charts/Test_Split/test_split_bar_chart.png" alt="Test Split Metrics Bar Chart" width="600" />

#### Test Split Confusion Matrices
<details>
<summary>Click to view Test Split Confusion Matrices</summary>

- <img src="./Output/Charts/Test_Split/test_split_SVM_RBF_cm.png" alt="SVM RBF" width="400" />
- <img src="./Output/Charts/Test_Split/test_split_SVM_Linear_cm.png" alt="SVM Linear" width="400" />
- <img src="./Output/Charts/Test_Split/test_split_Log_Regression_cm.png" alt="Logistic Regression" width="400" />
- <img src="./Output/Charts/Test_Split/test_split_Random_Forest_cm.png" alt="Random Forest" width="400" />
- <img src="./Output/Charts/Test_Split/test_split_Bagging_cm.png" alt="Bagging" width="400" />
- <img src="./Output/Charts/Test_Split/test_split_AdaBoost_cm.png" alt="AdaBoost" width="400" />
</details>

*(Validation and Test Split distributions generally reflect similar high accuracy, confirming model stability prior to final phase evaluation).*

---

### Final Test Evaluation (29 Completely Unseen Images)
These are 29 specifically selected test images (one for each class) independent of the primary training dataset.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest** | 100.00% | 100.00% | 100.00% | 100.00% |
| **Log Regression** | 100.00% | 100.00% | 100.00% | 100.00% |
| **SVM Linear** | 100.00% | 100.00% | 100.00% | 100.00% |
| **SVM RBF** | 100.00% | 100.00% | 100.00% | 100.00% |
| **AdaBoost** | 60.71% | 54.60% | 58.62% | 55.75% |
| **Bagging** | 100.00% | 100.00% | 100.00% | 100.00% |

<img src="./Output/Charts/Final_29/final_29_test_bar_chart.png" alt="Final Test Metrics Bar Chart" width="600" />

#### Final Test Confusion Matrices
<details>
<summary>Click to view Final Test Confusion Matrices</summary>

- <img src="./Output/Charts/Final_29/final_29_test_SVM_RBF_cm.png" alt="SVM RBF" width="400" />
- <img src="./Output/Charts/Final_29/final_29_test_SVM_Linear_cm.png" alt="SVM Linear" width="400" />
- <img src="./Output/Charts/Final_29/final_29_test_Log_Regression_cm.png" alt="Logistic Regression" width="400" />
- <img src="./Output/Charts/Final_29/final_29_test_Random_Forest_cm.png" alt="Random Forest" width="400" />
- <img src="./Output/Charts/Final_29/final_29_test_Bagging_cm.png" alt="Bagging" width="400" />
- <img src="./Output/Charts/Final_29/final_29_test_AdaBoost_cm.png" alt="AdaBoost" width="400" />
</details>

---

## 🏆 Model Configuration & Performance Analysis

This project benchmarked several classical machine learning algorithms without relying on Deep Neural Networks (CNNs). The model hyperparameters were explicitly constrained (e.g., `n_jobs=1`, restricted tree depths, reduced estimators) to operate effectively under strict pure-CPU memory limits within Google Colab. 

Here is a breakdown of the specific techniques used for each model and a comparative performance analysis:

### 1. **Support Vector Machines (SVM: Linear & RBF)**
- **Techniques Used:** Margin maximization. The Linear kernel creates strict hyperplanes, while RBF (Radial Basis Function) handles non-linear relationships using the kernel trick in a higher-dimensional space. `C=1.0` was kept standard.
- **Performance:** **👑 Best Overall.** Both SVM variants achieved near-perfect accuracy (~99.7% Validation, ~99.8% Test Split, and 100% Final Test). The dataset, after PCA reduced it to retain 95% variance, formed highly distinct clusters. SVM is mathematically unmatched at drawing optimized decision boundaries through such distinct, high-dimensional spaces.

### 2. **Logistic Regression**
- **Techniques Used:** Probabilistic linear classification using the sigmoid function (`max_iter` extended to 1000 for complex convergence).
- **Performance:** **Excellent.** Logistic Regression achieved 99.55% accuracy on Validation and 100% on the Final Test. Since the PCA feature matrix was well-scaled (`StandardScaler`) and highly linearly separable post-feature engineering, this simple model was surprisingly robust and efficient.

### 3. **Random Forest**
- **Techniques Used:** Ensemble learning using an aggregation of 100 simple decision trees. Utilizes bagging and feature randomness to prevent overfitting.
- **Performance:** **Excellent.** Reached 99.37% on Validation and 100% on the Final Test. Random Forest naturally distributes weights smoothly across complex, high-dimensional data, making it perfect for reading our combined global image features + local Bag-of-Visual-Words histograms.

### 4. **Bagging Classifier**
- **Techniques Used:** Bootstrap Aggregating over manually restricted Decision Trees (`max_depth=15`, 50 estimators, sampling 80% of features/samples). Built to reduce model variance sequentially. 
- **Performance:** **Good.** Maintained roughly ~94.5% across splits and hit 100% on the tiny 29-image final test set. While highly performant, it lost a slight edge compared to Random Forest due to the strict artificial CPU limits imposed on its underlying trees to purposely prevent RAM overflow during training.

### 5. **AdaBoost**
- **Techniques Used:** Adaptive Boosting using very shallow trees (`max_depth=5`) sequentially forced to focus and penalize previously misclassified training samples.
- **Performance:** **⚠️ Worst Overall.** Averaged around 54% to 60% accuracy. Because AdaBoost was built with weak learners (`max_depth=5` and only 50 estimators strictly enforced to save memory footprint), it simply lacked the representational depth needed to classify 29 distinct, nuanced ASL hand configurations accurately over tens of thousands of samples.

---

## ⚙️ Running the Project

1. Upload the `AIProjectML.ipynb` directly to Google Colab.
2. The project will automatically:
   - Mount your Google Drive.
   - Unzip the dataset seamlessly into the local Colab storage for faster processing.
   - Run feature extraction sequentially to ensure CPU RAM is well optimized.
   - Perform model training and output the artifacts above directly into the `/Output/` directory within your Drive.
