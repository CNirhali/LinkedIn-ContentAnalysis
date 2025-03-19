import joblib
import logging
from datetime import datetime


def save_model(model, vectorizer, version=None):
    # Generate version identifier if not provided
    if not version:
        version = datetime.now().strftime("%Y%m%d%H%M%S")

    # Save model and vectorizer
    joblib.dump(model, f"content_safety_model_v{version}.pkl")
    joblib.dump(vectorizer, f"content_vectorizer_v{version}.pkl")

    print(f"Model and vectorizer saved with version: {version}")
    return version


def load_model(version):
    # Load model and vectorizer
    model = joblib.load(f"content_safety_model_v{version}.pkl")
    vectorizer = joblib.load(f"content_vectorizer_v{version}.pkl")

    return model, vectorizer


def setup_monitoring(log_file="safety_algorithm_logs.txt"):
    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Example log entries
    logging.info("Safety algorithm monitoring initialized")

    def log_prediction(content_id, prediction, actual=None):
        """Log each prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'content_id': content_id,
            'predicted': prediction['is_harmful'],
            'confidence': prediction['confidence'],
            'actual': actual
        }
        logging.info(f"Prediction: {log_entry}")

    return log_prediction
