import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the extracted features dataset
file_path = 'E:/Final year project/Fall Classificaiton/Fall_Classification/Ankle/Extracted_Features_Grouped.csv'
data = pd.read_csv(file_path)

# Define feature columns and target
feature_columns = [col for col in data.columns if col not in ['label', 'Participant_ID', 'Trial_Number']]
X = data[feature_columns]
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel='rbf', C=1, gamma='scale', random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

# Train and evaluate models
for model_name, model in models.items():
    print(f"nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    print(f"/nMetrics for {model_name}:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy for {model_name}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
