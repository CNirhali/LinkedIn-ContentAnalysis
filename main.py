def build_safety_pipeline(training_data=None, model_version=None):
    """
    Build and return a complete content safety pipeline.
    If training_data is provided, trains a new model.
    If model_version is provided, loads an existing model.
    """
    if training_data and not model_version:
        # Extract features from training data
        features, vectorizer = extract_features(training_data['content'])
        labels = training_data['labels']

        # Train and fine-tune model
        model = train_content_classifier(features, labels)
        optimized_model = fine_tune_model(features, labels)

        # Save model
        version = save_model(optimized_model, vectorizer)

    elif model_version:
        # Load existing model
        model, vectorizer = load_model(model_version)

    else:
        raise ValueError("Either training_data or model_version must be provided")

    # Set up monitoring
    log_prediction = setup_monitoring()

    # Return a function that processes new content
    def process_content(content_items):
        # Prioritize content
        prioritized_queue = prioritize_content_queue(content_items, model, vectorizer)

        # Log predictions for monitoring
        for priority_level in prioritized_queue.values():
            for item in priority_level:
                log_prediction(item['content_id'], {
                    'is_harmful': item['is_harmful'],
                    'confidence': item['confidence']
                })

        return prioritized_queue

    return process_content
