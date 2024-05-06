import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('korean_drama.csv')

# Initialize SnowballStemmer for stemming
stemmer = SnowballStemmer('english')

# Tokenize the reviews into sentences
tokenized_sentences = []
for review in df['synopsis']:
    sentences = sent_tokenize(str(review))
    tokenized_sentences.extend(sentences)

# Tokenize words, perform stemming, and store the results
stemmed_feedback = []
for sentence in tokenized_sentences:
    words = word_tokenize(sentence)
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_sentence = ' '.join(stemmed_words)
    stemmed_feedback.append(stemmed_sentence)

# Print the first 10 tokenized and stemmed sentences
for i in range(10):
    print(stemmed_feedback[i])
