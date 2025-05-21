import logging
from typing import Dict, List, Any, Optional
from model import ContentSafetyModel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContentSafetyPipeline:
    def __init__(self):
        self.model = ContentSafetyModel()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.monitoring_enabled = False

    def build_pipeline(self, training_data: Optional[Dict[str, Any]] = None, model_version: Optional[str] = None):
        """
        Build and return a complete content safety pipeline.
        
        Args:
            training_data: Dictionary containing 'content' and 'labels' for training
            model_version: Path to existing model version to load
        """
        try:
            if training_data and not model_version:
                logger.info("Training new model with provided data")
                # Extract features from training data
                features = self.vectorizer.fit_transform(training_data['content']).toarray()
                labels = np.array(training_data['labels'])

                # Train model
                accuracy, metrics = self.model.train(features, labels)
                logger.info(f"Model training completed with accuracy: {accuracy:.2f}")

            elif model_version:
                logger.info(f"Loading existing model from version: {model_version}")
                self.model.load(f"{model_version}_model.joblib", f"{model_version}_vectorizer.joblib")

            else:
                raise ValueError("Either training_data or model_version must be provided")

            # Set up monitoring
            self._setup_monitoring()

            return self.process_content

        except Exception as e:
            logger.error(f"Error building pipeline: {str(e)}")
            raise

    def process_content(self, content_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a list of content items and return prioritized queue.
        
        Args:
            content_items: List of dictionaries containing content to process
            
        Returns:
            Dictionary with priority levels as keys and lists of processed items as values
        """
        try:
            prioritized_queue = {
                'high': [],
                'medium': [],
                'low': []
            }

            for item in content_items:
                # Get prediction
                prediction = self.model.predict(item['content'])
                
                # Determine priority based on prediction
                priority = self._determine_priority(prediction)
                
                # Add to appropriate queue
                processed_item = {
                    'content_id': item['content_id'],
                    'content': item['content'],
                    'is_harmful': prediction['is_harmful'],
                    'confidence': prediction['confidence'],
                    'severity_score': prediction['severity_score']
                }
                
                prioritized_queue[priority].append(processed_item)
                
                # Log prediction if monitoring is enabled
                if self.monitoring_enabled:
                    self._log_prediction(processed_item)

            return prioritized_queue

        except Exception as e:
            logger.error(f"Error processing content: {str(e)}")
            raise

    def _determine_priority(self, prediction: Dict[str, Any]) -> str:
        """Determine priority level based on prediction results."""
        if prediction['is_harmful'] and prediction['confidence'] > 0.8:
            return 'high'
        elif prediction['is_harmful'] or prediction['confidence'] > 0.6:
            return 'medium'
        return 'low'

    def _setup_monitoring(self):
        """Set up monitoring for the pipeline."""
        self.monitoring_enabled = True
        logger.info("Monitoring enabled for content safety pipeline")

    def _log_prediction(self, item: Dict[str, Any]):
        """Log prediction results for monitoring."""
        logger.info(
            f"Content ID: {item['content_id']}, "
            f"Harmful: {item['is_harmful']}, "
            f"Confidence: {item['confidence']:.2f}, "
            f"Severity: {item['severity_score']:.2f}"
        )
