import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from typing import Dict, Tuple, Any
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentSafetyModel:
    def __init__(self, model_config: Dict[str, Any] = None):
        self.model_config = model_config or {
            'objective': 'binary:logistic',
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.model = None
        self.vectorizer = None

    def train(self, features: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Train the content safety model with proper error handling and logging.
        """
        try:
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )

            # Initialize and train XGBoost classifier
            self.model = xgb.XGBClassifier(**self.model_config)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=False
            )

            # Evaluate the model
            y_pred = self.model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)
            
            logger.info(f"Model training completed. Accuracy: {metrics['accuracy']:.2f}")
            return metrics['accuracy'], metrics

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def predict(self, content: str) -> Dict[str, Any]:
        """
        Make predictions with proper error handling and input validation.
        """
        if not self.model or not self.vectorizer:
            raise ValueError("Model or vectorizer not initialized. Please train the model first.")

        try:
            # Preprocess and transform content
            processed_content = self._preprocess_text(content)
            content_features = self.vectorizer.transform([processed_content]).toarray()

            # Add additional features
            content_length = len(content)
            word_count = len(processed_content.split())

            # Combine features
            full_features = np.concatenate([
                content_features,
                np.array([[content_length, word_count]])
            ], axis=1)

            # Make prediction
            prediction = self.model.predict(full_features)[0]
            probability = self.model.predict_proba(full_features)[0, 1]

            return {
                'is_harmful': bool(prediction),
                'confidence': float(probability),
                'severity_score': float(probability * 100)
            }

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def save(self, model_path: str, vectorizer_path: str):
        """Save model and vectorizer to disk."""
        try:
            joblib.dump(self.model, model_path)
            joblib.dump(self.vectorizer, vectorizer_path)
            logger.info(f"Model saved successfully to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self, model_path: str, vectorizer_path: str):
        """Load model and vectorizer from disk."""
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    @staticmethod
    def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate and return model metrics."""
        report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'confusion_matrix': conf_matrix.tolist()
        }

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """Preprocess text for model input."""
        # Add your text preprocessing logic here
        return text.lower().strip()
