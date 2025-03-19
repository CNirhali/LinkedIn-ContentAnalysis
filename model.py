import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb


def train_content_classifier(features, labels):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Initialize and train XGBoost classifier
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model


def predict_content_safety(model, vectorizer, new_content):
    # Preprocess new content
    processed_content = preprocess_text(new_content)

    # Transform to feature vector
    content_features = vectorizer.transform([processed_content]).toarray()

    # Add additional features
    content_length = len(new_content)
    word_count = len(processed_content.split())

    # Combine all features
    full_features = np.concatenate([
        content_features,
        np.array([[content_length, word_count]])
    ], axis=1)

    # Make prediction
    prediction = model.predict(full_features)[0]
    probability = model.predict_proba(full_features)[0, 1]

    return {
        'is_harmful': bool(prediction),
        'confidence': float(probability),
        'severity_score': float(probability * 100)
    }
