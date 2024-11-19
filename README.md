# Cancer-Research-
tryıng dıfferent models on our exıstıng data of breast cancer 

websıte 
https://www.nature.com/articles/s41598-024-66658-x#Sec2

In summary, this study successfully developed and validated artificial intelligence clinical models and combined models using LightGBM machine learning algorithms based on clinical blood markers and ultrasound data to predict distant metastasis in breast cancer patients. Particularly, the combined model integrating clinical blood markers and ultrasound features exhibited high accuracy in predicting and identifying breast cancer distant metastasis, demonstrating potential clinical application value. These significant findings highlight the potential of developing economically efficient and easily obtainable predictive tools in clinical oncology. They are poised to elevate the level of clinical decision-making and prognosis assessment, potentially reducing the need for expensive or invasive imaging techniques. The research underscores the prospects of utilizing readily available clinical blood markers and cost-effective ultrasound data to develop predictive tools, holding critical significance for the advancement of clinical oncology, potentially offering patients more convenient and efficient healthcare.

import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load clinical data from Excel
clinical_data = pd.read_excel('path_to_clinical_data.xlsx')

# Function to load ultrasound images
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to a common shape if needed
    return image

# Example: Load an image
image = load_image('path_to_image.jpg')
# Extract relevant features (e.g., tumor markers, liver function) from clinical data
clinical_features = clinical_data[['carcinoembryonic_antigen', 'alpha_fetoprotein', 'CA125', 'CA153', 'CA199', 
                                   'total_bilirubin', 'direct_bilirubin', 'urea', 'creatinine', 'cholesterol']]  # Example columns

# Standardize the clinical features
scaler = StandardScaler()
clinical_features_scaled = scaler.fit_transform(clinical_features)
# Function to extract features from ultrasound image (e.g., lesion size)
def extract_image_feature(image):
    # Here we would extract the lesion diameter using image processing (e.g., edge detection)
    # Example: Convert to grayscale and detect contours (simplified)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_diameter = 0
    
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        max_diameter = max(max_diameter, radius)
    
    return max_diameter

# Example: Extract feature from ultrasound image
lesion_diameter = extract_image_feature(image)
from sklearn.linear_model import LassoCV
from scipy.stats import spearmanr

# Calculate Spearman's rank correlation
corr_matrix = np.corrcoef(clinical_features_scaled.T)
spearman_corr, _ = spearmanr(clinical_features_scaled)

# Feature selection using LASSO
lasso = LassoCV(cv=5)
lasso.fit(clinical_features_scaled, clinical_data['target'])  # 'target' is the column with labels
selected_features = lasso.coef_  # Features with non-zero coefficients

# Print selected features
print(selected_features)
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score

# Prepare data (use selected features)
X = clinical_features_scaled  # Include image features if needed
y = clinical_data['target']   # Target variable

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM model
params = {'objective': 'binary', 'metric': 'auc'}
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Train the model
clf = lgb.train(params, train_data, 100, valid_sets=[test_data], early_stopping_rounds=10)

# Predict
y_pred = clf.predict(X_test, num_iteration=clf.best_iteration)

# Evaluate performance (e.g., AUC, accuracy)
from sklearn.metrics import roc_auc_score, accuracy_score
print(f'AUC: {roc_auc_score(y_test, y_pred)}')
print(f'Accuracy: {accuracy_score(y_test, (y_pred > 0.5))}')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

