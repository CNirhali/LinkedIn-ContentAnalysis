# LinkedIn Content Safety Analysis

A robust machine learning pipeline for analyzing and classifying content safety in LinkedIn posts using XGBoost and advanced NLP techniques.

## 🚀 Features

- **Advanced Content Classification**: Utilizes XGBoost for high-accuracy content safety classification
- **Real-time Processing**: Efficient pipeline for processing content in real-time
- **Priority-based Queue**: Intelligent content prioritization based on safety scores
- **Comprehensive Monitoring**: Built-in logging and monitoring system
- **Model Persistence**: Save and load trained models for future use
- **Type Safety**: Full type hints for better code maintainability
- **Error Handling**: Robust error handling and logging throughout the pipeline

## 🛠️ Technical Stack

- Python 3.8+
- XGBoost for machine learning
- scikit-learn for feature extraction and model evaluation
- NumPy for numerical operations
- TF-IDF Vectorization for text processing
- Joblib for model persistence

## 📋 Prerequisites

```bash
pip install -r requirements.txt
```

## 🏗️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LinkedIn-ContentAnalysis.git
cd LinkedIn-ContentAnalysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Usage

```python
from main import ContentSafetyPipeline

# Initialize the pipeline
pipeline = ContentSafetyPipeline()

# Train a new model
pipeline.build_pipeline(training_data={
    'content': your_content_list,
    'labels': your_labels_list
})

# Process content
results = pipeline.process_content([
    {'content_id': '1', 'content': 'some text to analyze'},
    {'content_id': '2', 'content': 'more text to analyze'}
])
```

### Loading Existing Model

```python
pipeline = ContentSafetyPipeline()
pipeline.build_pipeline(model_version='path_to_model')
```

## 📊 Model Details

The content safety model uses the following features:
- TF-IDF vectorization with n-grams (1-2)
- Content length
- Word count
- Custom text preprocessing

### Priority Levels

Content is classified into three priority levels:
- **High**: Harmful content with confidence > 0.8
- **Medium**: Harmful content or confidence > 0.6
- **Low**: All other content

## 🔧 Configuration

The model can be configured through the `model_config` parameter:

```python
model_config = {
    'objective': 'binary:logistic',
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
```

## 📈 Performance Metrics

The model provides the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## 🔍 Monitoring

The pipeline includes built-in monitoring with detailed logging:
- Content ID tracking
- Harmful content detection
- Confidence scores
- Severity scores

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- LinkedIn for the inspiration
- XGBoost team for the amazing library
- scikit-learn team for the machine learning tools
