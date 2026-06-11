import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("dataset.csv")

# -------------------------
# Features & Target
# -------------------------

X = df.drop("output", axis=1)
y = df["output"]

# -------------------------
# Split Data
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# Train Model
# -------------------------

model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# Evaluation
# -------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================================================
# Get New Patient Information From User
# =====================================================

print("\n")
print("=" * 50)
print("ENTER PATIENT INFORMATION")
print("=" * 50)

age = int(input("Age: "))
sex = int(input("Sex (0=Female, 1=Male): "))
cp = int(input("Chest Pain Type (0-3): "))
trtbps = int(input("Resting Blood Pressure: "))
chol = int(input("Cholesterol: "))
fbs = int(input("Fasting Blood Sugar (0/1): "))
restecg = int(input("Rest ECG (0-2): "))
thalachh = int(input("Maximum Heart Rate: "))
exng = int(input("Exercise Angina (0/1): "))
oldpeak = float(input("Old Peak: "))
slp = int(input("Slope (0-2): "))
caa = int(input("Number of Major Vessels (0-3): "))
thall = int(input("Thal (0-3): "))

# -------------------------
# Create DataFrame
# -------------------------

new_patient = pd.DataFrame([[
    age,
    sex,
    cp,
    trtbps,
    chol,
    fbs,
    restecg,
    thalachh,
    exng,
    oldpeak,
    slp,
    caa,
    thall
]], columns=X.columns)

# -------------------------
# Prediction
# -------------------------

prediction = model.predict(new_patient)[0]

probability = model.predict_proba(new_patient)[0]

print("\n")
print("=" * 50)
print("RESULT")
print("=" * 50)

if prediction == 1:
    print("High Risk Of Heart Attack")
else:
    print("Low Risk Of Heart Attack")

print(f"\nProbability Low Risk : {probability[0]*100:.2f}%")
print(f"Probability High Risk: {probability[1]*100:.2f}%")