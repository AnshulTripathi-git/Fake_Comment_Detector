import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv('test_data.csv')

# Use the correct text and label columns
df = df.rename(columns={'text_': 'review', 'label': 'label_raw'})

# Map label to binary: 0 = genuine, 1 = fake
# You may need to adjust this mapping based on your dataset's label meanings
df['label'] = df['label_raw'].map({'CG': 1, 'OR': 0})

# Drop rows with missing values in review or label
df = df.dropna(subset=['review', 'label'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
)

# Vectorize text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Example usage
def predict_review(review_text):
    vec = vectorizer.transform([review_text])
    pred = model.predict(vec)[0]
    return "Fake" if pred == 1 else "Genuine"

# Test prediction
print(predict_review("This product is amazing, I love it!"))
print(predict_review("Fake review, do not trust!"))