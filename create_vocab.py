import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
from torchtext.vocab import vocab

from load_data import questions
from processing import preprocess_text


max_len = 98
processed_questions = [preprocess_text(q) for q in questions]

# Tạo từ điển vocab
def build_vocab(tokenized_texts):
    word_freq = Counter([word for text in tokenized_texts for word in text])
    word_vocab = vocab(word_freq, specials=["<pad>", "<unk>"])
    word_vocab.set_default_index(word_vocab["<unk>"])
    return word_vocab

word_vocab = build_vocab(processed_questions)
def numericalize(text, vocab):
    return [vocab[word] for word in text]

# Padding dữ liệu
def pad_sequences(sequences, max_len):
    return [seq + [word_vocab["<pad>"]] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences]
numerical_questions = [numericalize(q, word_vocab) for q in processed_questions]
numerical_questions = pad_sequences(numerical_questions, max_len)
numerical_questions = torch.tensor(numerical_questions)