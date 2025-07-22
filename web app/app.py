from flask import Flask, render_template, request
import torch
import torch.nn as nn
import joblib
import numpy as np

app = Flask(__name__)

# Load vectorizer and label encoder
vectorizer = joblib.load("../tfidf_vectorizer.pkl")
label_encoder = joblib.load("../label_encoder.pkl")

# Define model
class MentalHealthClassifier(nn.Module):
    def __init__(self, input_size=5000, hidden_size=128, num_classes=3):
        super(MentalHealthClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model
model = MentalHealthClassifier()
model.load_state_dict(torch.load("../models/mental_health_model.pth", map_location=torch.device("cpu")))
model.eval()

# Prediction function
def predict(post):
    processed = post.lower()
    X = vectorizer.transform([processed]).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        output = model(X_tensor)
        probs = torch.softmax(output, dim=1).numpy().flatten()
        pred_class = np.argmax(probs)
        label = label_encoder.inverse_transform([pred_class])[0]

    # Emoji and color mapping
    emoji_map = {
        "Normal": "🟢",
        "At-Risk": "⚠️",
        "Needs Immediate Attention": "🔴"
    }
    color_map = {
        "Normal": "green",
        "At-Risk": "yellow",
        "Needs Immediate Attention": "red"
    }

    # Most influential words
    top_indices = np.argsort(X[0])[::-1][:5]
    impactful_words = [vectorizer.get_feature_names_out()[i] for i in top_indices if X[0][i] > 0]

    return {
        "label": label,
        "emoji": emoji_map[label],
        "color": color_map[label],
        "probs": dict(zip(label_encoder.classes_, probs.round(4))),
        "impactful_words": impactful_words
    }

# Web route
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        user_input = request.form["post"]
        if user_input.strip():
            result = predict(user_input)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
