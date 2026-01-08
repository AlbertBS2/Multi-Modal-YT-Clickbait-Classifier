"""
NLP Feature Extraction Module for Clickbait Detection

This module provides functions to extract various NLP features from video transcripts,
including sentence embeddings, statistical features, sentiment features, and linguistic complexity metrics.
"""

import re
import numpy as np
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords


# STATISTICAL ANALYSIS
def extract_statistical_features(transcript):
    """
    Extract statistical text features from a transcript.

    Args:
        transcript (str): The transcript text to analyze.

    Returns:
        features (np.ndarray): 15-dimensional array of statistical features.
    """

    # EDGE CASE
    if not transcript or len(transcript.strip()) == 0:
        return np.zeros(15)

    # BASIC COUNTS
    transcript_length = len(transcript)
    words = word_tokenize(transcript.lower())
    word_count = len(words)

    # AVOID DIVISION BY ZERO
    if word_count == 0:
        return np.zeros(15)

    # AVERAGE WORD LENGTH
    avg_word_length = np.mean([len(word) for word in words])

    # SENTENCE STATISTICS
    sentences = sent_tokenize(transcript)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / max(sentence_count, 1)

    # PUNCTUATION COUNTS
    exclamation_count = transcript.count('!')
    question_count = transcript.count('?')
    ellipsis_count = transcript.count('...')

    # CAPITALIZATION RATIOS
    words_text = re.findall(r'\b[A-Za-z]+\b', transcript)
    if len(words_text) == 0:
        uppercase_word_ratio = 0
        capitalized_word_ratio = 0
        all_caps_ratio = 0
    else:
        uppercase_words = [w for w in words_text if w[0].isupper()]
        all_caps_words = [w for w in words_text if w.isupper() and len(w) > 1]

        uppercase_word_ratio = len(uppercase_words) / len(words_text)

        # CAPITALIZED WORDS EXCLUDING ALL SENTENCE STARTS
        capitalized_word_ratio = (len(uppercase_words) - sentence_count) / max(len(words_text), 1)
        all_caps_ratio = len(all_caps_words) / len(words_text)

    # NUMBER COUNT
    number_count = len(re.findall(r'\b\d+\b', transcript))

    # VOCABULARY DIVERSITY
    unique_words = set(words)
    unique_word_ratio = len(unique_words) / word_count

    # STOPWORD RATIO
    try:
        stop_words = set(stopwords.words('english'))
        stopword_count = len([w for w in words if w in stop_words])
        stopword_ratio = stopword_count / word_count
    except:
        stopword_ratio = 0

    # PUNCTUATION DENSITY
    punctuation_count = len(re.findall(r'[^\w\s]', transcript))
    punctuation_density = punctuation_count / transcript_length

    features = np.array([
        transcript_length,
        word_count,
        avg_word_length,
        sentence_count,
        avg_sentence_length,
        exclamation_count,
        question_count,
        uppercase_word_ratio,
        capitalized_word_ratio,
        number_count,
        unique_word_ratio,
        stopword_ratio,
        punctuation_density,
        all_caps_ratio,
        ellipsis_count
    ])

    return features


# SENTIMENT ANALYSIS
def extract_sentiment_features(transcript):
    """
    Extract sentiment features from a transcript using TextBlob.

    Args:
        transcript (str): The transcript text to analyze.

    Returns:
        features (np.ndarray): 3-dimensional array of sentiment features.
    """

    # EDGE CASE
    if not transcript or len(transcript.strip()) == 0:
        return np.zeros(3)

    blob = TextBlob(transcript)

    # POLARITY: -1 (negative) to 1 (positive)
    polarity = blob.sentiment.polarity

    # SUBJECTIVITY: 0 (objective) to 1 (subjective)
    subjectivity = blob.sentiment.subjectivity

    # INTENSITY (absolute value of polarity)
    intensity = abs(polarity)

    features = np.array([polarity, subjectivity, intensity])

    return features


# LANGUAGE COMPLEXITY ANALYSIS
def extract_complexity_features(transcript):
    """
    Extract linguistic complexity features from a transcript.

    Args:
        transcript (str): The transcript text to analyze.

    Returns:
        features (np.ndarray): 4-dimensional array of complexity features.
    """

    # EDGE CASE
    if not transcript or len(transcript.strip()) == 0:
        return np.zeros(4)

    # TOKENIZE AND COUNT WORDS
    words = word_tokenize(transcript.lower())
    word_count = len(words)

    # CASE
    if word_count == 0:
        return np.zeros(4)

    # LEXICAL DENSITY: RATIO OF CONTENT WORDS TO TOTAL WORDS
    try:
        stop_words = set(stopwords.words('english'))
        content_words = [w for w in words if w.isalpha() and w not in stop_words]
        lexical_density = len(content_words) / word_count
    except:
        lexical_density = 0

    # AVERAGE SYLLABLES PER WORD (SIMPLIFIED ESTIMATION)
    def count_syllables(word):
        """Estimate syllable count based on vowel groups."""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Adjust for silent 'e'
        if word.endswith('e'):
            syllable_count -= 1

        # Ensure at least one syllable
        if syllable_count == 0:
            syllable_count = 1

        return syllable_count

    syllables = [count_syllables(w) for w in words if w.isalpha()]
    avg_syllables_per_word = np.mean(syllables) if syllables else 0

    # FLESCH READING EASE SCORE (COMBINATION OF AVERAGE LENGTH AND AVERAGE SYLLABLES PER WORD)
    sentences = sent_tokenize(transcript)
    sentence_count = len(sentences)
    total_syllables = sum(syllables)

    if sentence_count > 0 and word_count > 0:
        flesch_reading_ease = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (total_syllables / word_count)
    else:
        flesch_reading_ease = 0

    # AUTOMATED READABILITY INDEX (ARI)
    if sentence_count > 0 and word_count > 0:
        characters = sum(len(w) for w in words)
        automated_readability_index = 4.71 * (characters / word_count) + 0.5 * (word_count / sentence_count) - 21.43
    else:
        automated_readability_index = 0

    features = np.array([
        lexical_density,
        avg_syllables_per_word,
        flesch_reading_ease,
        automated_readability_index
    ])

    return features


# EMBEDDING ANALYSIS
def extract_sentence_embeddings(transcript, model):
    # EDGE CASE
    if not transcript or len(transcript.strip()) == 0:
        return np.zeros(384)

    # CREATE LIST OF WORDS
    words = transcript.split()

    # SPLIT INTO CHUNKS OF 100 WORDS
    chunks = [" ".join(words[i: i + 100])
              for i in range(0, len(words), 100)]

    # ENCODE ALL CHUNKS
    embeddings = model.encode(chunks, show_progress_bar=False)

    # AVERAGE EMBEDDINGS FOR THE TRANSCRIPT
    master_embedding = np.mean(embeddings, axis=0)

    return master_embedding


def extract_all_nlp_features(transcript, model):
    """
    Extract all NLP features from a transcript (statistical, sentiment, complexity, and embeddings).

    Args:
        transcript (str): The transcript text to analyze.
        model (SentenceTransformer): Pre-trained sentence transformer model for embeddings.

    Returns:
        all_features (np.ndarray): 406-dimensional feature vector
                                    (384 embeddings + 15 statistical + 3 sentiment + 4 complexity).
    """
    # Extract all feature types
    statistical = extract_statistical_features(transcript)
    sentiment = extract_sentiment_features(transcript)
    complexity = extract_complexity_features(transcript)
    embeddings = extract_sentence_embeddings(transcript, model)

    # Concatenate all features
    all_features = np.concatenate([embeddings, statistical, sentiment, complexity])

    return all_features
