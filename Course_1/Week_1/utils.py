import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def process_review(review):
    """
    Process review function.
    Input:
        review: a string containing a review
    Output:
        review_clean: a list of word containing a processed review
    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove hyperlinks
    review = re.sub(r'https?:\/\/.*[\r\n]*', '', review)
    # remove hashtags
    # only removing the hash # sign from the word
    review = re.sub(r'#', '', review)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    review_tokens = tokenizer.tokenize(review)

    review_clean = []
    for word in review_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            review_clean.append(stem_word)

    return review_clean

def build_freqs(reviews):
    """
    Build frequencies.
    Input:
        reviews: a list of reviews
        ys: an m x 1 array with the sentiment label of each review
        (either 0 or 1)
   Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its frequency
   """
    
    # Start with an empty dictionary and populate it by looping over all the reviews
    # and over all processed words in each review
    freqs = {}

    def append_freqs(x):
        for word in process_review(x['review_body']): 
            pair = (word, x['sentiment'])
            freqs[pair] = freqs.get(pair,0) + 1
            
    reviews.foreach(lambda x: append_freqs(x))
    return freqs


    