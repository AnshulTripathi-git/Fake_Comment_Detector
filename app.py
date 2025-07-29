from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load and prepare data
df = pd.read_csv('test_data.csv')
df = df.rename(columns={'text_': 'review', 'label': 'label_raw'})
df['label'] = df['label_raw'].map({'CG': 1, 'OR': 0})
df = df.dropna(subset=['review', 'label'])

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['label']
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

def predict_review(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "Fake" if pred == 1 else "Genuine"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    comment = ""
    if request.method == 'POST':
        comment = request.form['comment']
        result = predict_review(comment)
    return render_template('index.html', result=result, comment=comment)

if __name__ == '__main__':
    app.run(debug=True)