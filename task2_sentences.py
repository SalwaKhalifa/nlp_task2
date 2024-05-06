import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('korean_drama.csv')

# Initialize SnowballStemmer for stemming
stemmer = SnowballStemmer('english')

# Tokenize and stem the reviews
def tokenize_and_stem(text):
    tokenized_sentences = sent_tokenize(str(text))
    stemmed_feedback = []
    for sentence in tokenized_sentences:
        words = word_tokenize(sentence)
        stemmed_words = [stemmer.stem(word) for word in words]
        stemmed_sentence = ' '.join(stemmed_words)
        stemmed_feedback.append(stemmed_sentence)
    return stemmed_feedback

# Apply tokenization and stemming to each synopsis
df['stemmed_synopsis'] = df['synopsis'].apply(tokenize_and_stem)

# Print the first 10 tokenized and stemmed sentences
for i in range(10):
    print(df['stemmed_synopsis'][i])
