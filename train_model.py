import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

y = df['diagnosis'].map({'B': 0, 'M': 1})
X = df.drop(columns=['diagnosis'])
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "tumor_model.sav")
print("✅ Model trained and saved as tumor_model.sav")
