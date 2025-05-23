import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import numpy as np
from collections import Counter
import joblib

# 1. Loading the dataset
df = pd.read_csv(r"Insert dataset for training")
df = df.drop(columns=["Window", "MMSI"], errors="ignore")
df = df.dropna()

# 2. Drop irrelevant columns and target Behavior_Label
X = df.drop(columns=["Behavior_Label", "max_SOG", "min_SOG", "delta_SOG", 
                     "mean_abs_acceleration", "mean_acceleration", 
                     "mean_turning_intensity", "turning_spike_count", 
                     "far_from_coast"])
y = df["Behavior_Label"]

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Calculate sample weights for imbalanced classes
class_counts = Counter(y_train)
total = sum(class_counts.values())
class_weights = {cls: total / count for cls, count in class_counts.items()}
sample_weights = np.array([class_weights[label] for label in y_train])

# 5. Train XGBoost model hyperparameters found via grid search
model = XGBClassifier(
    objective="multi:softmax",
    num_class=len(y.unique()),
    eval_metric="mlogloss",
    learning_rate=0.1,
    n_estimators=300,
    max_depth=6,
    subsample=0.7,
    colsample_bytree=0.9,
    min_child_weight=10,
    gamma=0.1,
    random_state=42
)

# Hyperparameters found via trial and error and for generalization purpose
#model = XGBClassifier(
    #objective="multi:softmax",
    #num_class=len(y.unique()),
    #eval_metric="mlogloss",
    #learning_rate=0.1,
    #n_estimators=350,
    #max_depth=4,
    #subsample=0.6,
    #colsample_bytree=0.6,
    #min_child_weight=75,
    #gamma=0.1,
    #random_state=42
#)

model.fit(X_train, y_train, sample_weight=sample_weights)

# 6. Evaluation of model
y_pred = model.predict(X_test)

# Print metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))

# Udskriv rapport for hver klasse
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model to a joblib file
joblib.dump(model, "XGBoost_model_40min.joblib")