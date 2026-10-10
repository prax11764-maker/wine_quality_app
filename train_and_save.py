"""
train_and_save.py

Trains the final Wine Quality model with class-imbalance handling and saves
everything the Streamlit app needs: the trained model, the scaler, and the
label encoder. Run this once before running the Streamlit app.
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42

# 1. Load data
df = pd.read_csv("winequality.csv")
X = df.drop("quality", axis=1)
y = df["quality"]

# 2. Encode quality (3-8) -> (0-5) for consistency with the rest of the project
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 3. Train/test split (stratified, same as before)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

# 4. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. THE IMPROVEMENT: class_weight='balanced'
#    This tells the model to pay MORE attention to rare classes (quality 3, 4, 8)
#    and LESS to common ones (5, 6), instead of treating every mistake equally.
#    It's a one-line fix for imbalance that needs no extra installs.
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)
model.fit(X_train_scaled, y_train)

# 6. Evaluate
preds = model.predict(X_test_scaled)
print("Test accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds, zero_division=0))

# 7. Save everything the app needs
joblib.dump(model, "wine_model.pkl")
joblib.dump(scaler, "wine_scaler.pkl")
joblib.dump(le, "wine_label_encoder.pkl")
joblib.dump(list(X.columns), "wine_feature_names.pkl")

print("\nSaved: wine_model.pkl, wine_scaler.pkl, wine_label_encoder.pkl, wine_feature_names.pkl")
