import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load our sample dataset
data = pd.read_csv("students.csv")

# Select input features
X = data[["study_hours", "attendance", "previous_score"]]

# Select the target we want to predict
y = data["performance"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create the machine-learning model
model = DecisionTreeClassifier(random_state=42)

# Train using only the training data
model.fit(X_train, y_train)

# Test on data the model did not train on
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Save the trained model
with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")
print("Model saved as student_model.pkl")