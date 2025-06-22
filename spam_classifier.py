import pandas as pd

# Load the dataset
data = pd.read_csv("spam.csv", encoding='ISO-8859-1')

# Keep only the required columns
data = data[['label', 'text']]
data = data.rename(columns={'text': 'message'})

# Show first few rows
print(data.head())

# Show dataset size
print("\nShape of cleaned dataset:", data.shape)
# Convert labels to binary: ham = 0, spam = 1
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Check for missing values
print("\nMissing values:\n", data.isnull().sum())
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize vectorizer
vectorizer = TfidfVectorizer(max_features=3000)

# Transform messages to vectors
X = vectorizer.fit_transform(data['message']).toarray()

# Labels (0 for ham, 1 for spam)
y = data['label'].values

# Show the shape of features and labels
print("\nVectorized data shape:", X.shape)
print("Labels shape:", y.shape)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Show accuracy and report
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Real-time testing
print("\n🔍 Test Your Own Message")
user_msg = input("Enter your message: ")

# Transform the message using same vectorizer
user_data = vectorizer.transform([user_msg])

# Predict and print result
prediction = model.predict(user_data)[0]
label_map = {0: "HAM", 1: "SPAM"}
print("\n This message is predicted as:", label_map[prediction])