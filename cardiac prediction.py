import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def generate_enhanced_data(n=1000):
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 90, size=n),
        'sex': np.random.randint(0, 2, size=n),
        'chest_pain_level': np.random.randint(0, 11, size=n),
        'symptoms_severity': np.random.randint(0, 11, size=n),
        'trestbps': np.random.randint(94, 200, size=n),
        'chol': np.random.randint(126, 564, size=n),
        'ldl': np.random.randint(50, 200, size=n),
        'hdl': np.random.randint(20, 100, size=n),
        'ecg_result': np.random.randint(0, 2, size=n),
        'thalach': np.random.randint(71, 202, size=n),
        'exang': np.random.randint(0, 2, size=n),
        'oldpeak': np.random.uniform(0.0, 6.2, size=n),
        'slope': np.random.randint(0, 3, size=n),
        'ca': np.random.randint(0, 4, size=n),
        'thal': np.random.randint(0, 3, size=n),
        'target': np.random.randint(0, 2, size=n)
    })
    return df

df = generate_enhanced_data()
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"✅ Model trained. Accuracy: {accuracy * 100:.2f}%")
