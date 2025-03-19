import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from flask import Flask, request, jsonify, render_template
import os

# Load a local LLM model for content moderation
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"  # Placeholder, replace with your fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Flask App for Interactive UI
app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Perform classification
    result = classifier(text)
    return jsonify({"text": text, "analysis": result})


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    with open("templates/index.html", "w") as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LinkedIn Safety Analyzer</title>
            <link rel="stylesheet" type="text/css" href="/static/styles.css">
            <script>
                async function analyzeText() {
                    let text = document.getElementById("textInput").value;
                    let response = await fetch("/analyze", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ text: text })
                    });
                    let result = await response.json();
                    document.getElementById("result").innerText = JSON.stringify(result, null, 2);
                }
            </script>
        </head>
        <body>
            <div class="container">
                <h1>LinkedIn Safety Analyzer</h1>
                <textarea id="textInput" rows="4" cols="50" placeholder="Enter text here..."></textarea><br>
                <button onclick="analyzeText()">Analyze</button>
                <pre id="result"></pre>
            </div>
        </body>
        </html>
        """)

    with open("static/styles.css", "w") as f:
        f.write("""
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            text-align: center;
            padding: 20px;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            display: inline-block;
        }
        textarea {
            width: 80%;
            padding: 10px;
            margin-top: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            margin-top: 10px;
            cursor: pointer;
            border-radius: 5px;
        }
        button:hover {
            background-color: #0056b3;
        }
        pre {
            text-align: left;
            background: #eee;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        """)

    app.run(debug=True)
