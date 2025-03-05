import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 📌 Load dataset
df = pd.read_csv("data/Crop_recommendation.csv")

# 📌 Encode categorical values
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])  # Encode crop names

# Save label encoding for later use in API
joblib.dump(label_encoder, "model/label_encoder.pkl")

# 📌 Features & Target
X = df.drop(columns=['label'])  # Features (remove target column)
y = df['label']  # Target (crop name)

# 📌 Split dataset into Train & Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📌 Save Model
joblib.dump(model, "model/crop_model.pkl")

print("✅ Model trained and saved successfully!")
