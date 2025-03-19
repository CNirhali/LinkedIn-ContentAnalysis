import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


def extract_features(data):
    # Create a DataFrame from collected data
    df = pd.DataFrame(data)

    # Preprocess text
    df['processed_content'] = df['content'].apply(preprocess_text)

    # Extract TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_features = vectorizer.fit_transform(df['processed_content'])

    # Extract additional features that may indicate harmful content
    df['content_length'] = df['content'].apply(len)
    df['word_count'] = df['processed_content'].apply(lambda x: len(x.split()))

    # Create feature matrix
    feature_matrix = pd.concat([
        pd.DataFrame(tfidf_features.toarray(), columns=vectorizer.get_feature_names_out()),
        df[['content_length', 'word_count']]
    ], axis=1)

    return feature_matrix, vectorizer
