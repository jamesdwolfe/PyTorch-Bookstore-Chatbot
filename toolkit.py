import nltk  # nltk.download('punkt')
import numpy as np


def tokenize(string):
    return nltk.word_tokenize(string)


def stem(string):
    return nltk.stem.porter.PorterStemmer().stem(string.lower())


def bag_of_words(tokenized_string, all_words):
    tokenized_string = [stem(word) for word in tokenized_string if word.isalpha() or word.isdigit()]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for index, word in enumerate(all_words):
        if word in tokenized_string:
            bag[index] = 1.0
    return bag
