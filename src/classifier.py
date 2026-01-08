import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("data/data.csv")
df = data  # Alias for data exploration

# Features and labels
X = data["description"]
y = data["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.3, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Model accuracy:", accuracy)


# Show the first 5 rows
print(df.head())

# Check column names
print(df.columns)

# See how many rows and columns
print(df.shape)

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing essential info (if any)
df = df.dropna(subset=['title', 'description'])

# Optional: reset index after dropping rows
df = df.reset_index(drop=True)

# Combine relevant columns into a single text column for classification
df['text_data'] = df['title'] + " " + df['description']

# Preview the new column
print(df['text_data'].head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize vectorizer
vectorizer = TfidfVectorizer(max_features=500)  # you can adjust max_features

# Fit and transform the text data
X = vectorizer.fit_transform(df['text_data'])

# Preview shape of X
print(X.shape)

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder

# Suppose you have a target column 'label' for job type or role
# # Encode target labels
# le = LabelEncoder()
# y = le.fit_transform(df['label'])

# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train classifier
# clf = LogisticRegression(max_iter=500)
# clf.fit(X_train, y_train)

# # Evaluate accuracy
# accuracy = clf.score(X_test, y_test)
# print(f"Classifier Accuracy: {accuracy:.2f}")

# # Example new job description
# new_job = ["Junior Machine Learning Engineer, Python, SQL, pandas"]

# # Vectorize the new job
# new_job_vector = vectorizer.transform(new_job)

# # Predict category
# predicted_category = clf.predict(new_job_vector)
# predicted_label = le.inverse_transform(predicted_category)

# print(f"Predicted Category: {predicted_label[0]}")

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder

# # Encode labels
# le = LabelEncoder()
# y = le.fit_transform(df['label'])  # make sure you have a 'label' column

# # Convert text to numeric features
# vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
# X = vectorizer.fit_transform(df['text_data'])

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train classifier
# clf = LogisticRegression(max_iter=1000)
# clf.fit(X_train, y_train)

# # Evaluate
# accuracy = clf.score(X_test, y_test)
# print("Classifier Accuracy:", accuracy)

# # Predict new job
# new_job = ["Junior Machine Learning Engineer, Python, SQL, pandas"]
# new_job_vec = vectorizer.transform(new_job)
# predicted_category = clf.predict(new_job_vec)
# predicted_label = le.inverse_transform(predicted_category)
# print(f"Predicted Category: {predicted_label[0]}")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Example data
data = {
    'text_data': [
        "Analyze data using Python and SQL",
        "Build ML models in Python",
        "Develop backend systems in Java",
        "Analyze datasets and predict outcomes",
        "Train classification models"
    ],
    'Category': [
        "Data Analyst",
        "ML Engineer",
        "Software Engineer",
        "Data Scientist",
        "ML Engineer"
    ]
}

df = pd.DataFrame(data)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['Category'])

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text_data'])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Accuracy
print("Classifier Accuracy:", clf.score(X_test, y_test))

# Test prediction
new_job = ["Junior ML Engineer, Python, SQL"]
pred = clf.predict(vectorizer.transform(new_job))
print("Predicted Category:", le.inverse_transform(pred)[0])

import joblib
joblib.dump(clf, 'clf_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')