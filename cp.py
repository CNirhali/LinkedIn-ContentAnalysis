def prioritize_content_queue(content_items, model, vectorizer):
    # Process all content items
    results = []

    for item in content_items:
        # Get prediction for each item
        prediction = predict_content_safety(model, vectorizer, item['content'])

        # Add original content and metadata to results
        results.append({
            'content_id': item.get('id', ''),
            'content': item['content'],
            'author': item.get('author', ''),
            'timestamp': item.get('timestamp', ''),
            'is_harmful': prediction['is_harmful'],
            'confidence': prediction['confidence'],
            'severity_score': prediction['severity_score']
        })

    # Sort queue by severity score (descending)
    prioritized_queue = sorted(results, key=lambda x: x['severity_score'], reverse=True)

    # Group into priority levels
    high_priority = [item for item in prioritized_queue if item['severity_score'] >= 70]
    medium_priority = [item for item in prioritized_queue if 30 <= item['severity_score'] < 70]
    low_priority = [item for item in prioritized_queue if item['severity_score'] < 30]

    return {
        'high_priority': high_priority,
        'medium_priority': medium_priority,
        'low_priority': low_priority
    }
