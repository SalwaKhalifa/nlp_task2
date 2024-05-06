import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('korean_drama.csv')

# Initialize SnowballStemmer for stemming
stemmer = SnowballStemmer('english')

# Tokenize words, perform stemming, and store the results
stemmed_review = []
for review in df['synopsis']:
    sentences = sent_tokenize(str(review))  # Tokenize sentences
    for sentence in sentences:
        words = word_tokenize(sentence)  # Tokenize words
        stemmed_words = [stemmer.stem(word) for word in words]  # Stem each word
        stemmed_review.append(stemmed_words)  # Append stemmed words to the list

# Print the first 10 tokenized and stemmed sentences
for i in range(10):
    print(stemmed_review[i])
