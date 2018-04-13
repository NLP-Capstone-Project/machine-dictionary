from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import math
import numpy as Numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from collections import Counter

SOS_token = 0
EOS_token = 1

hidden_size = 256

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.nwords = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        word.lower()
        if word not in self.word2index:
            self.word2index[word] = self.nwords
            self.index2word[self.nwords] = word
            self.nwords += 1

lang = Lang("English")

with open("pride.txt") as f:
    text = f.read()

sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)
for sentence in sentences:
    lang.addSentence(sentence)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.i2h = nn.Linear(hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, input_size)
        self.o2o = nn.Linear(hidden_size, input_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, word, hidden):
        embedded = self.embedding(word).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        newHidden = self.i2h(embedded + hidden)
        output = self.i2o(embedded + hidden)
        output = self.dropout(output)

        hidden = newHidden
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    return result

def firstWord(lang, word):
    indexes = indexesFromSentence(lang, word)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    return result

def train(input_variable, encoder, encoder_optimizer, criterion):
    # initialize the hidden states of the encoder
    hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    input_len = input_variable.size()[0]
    loss = Variable(torch.zeros(1))
    for i in range(input_len):
        output, hidden = encoder(
            input_variable[i], hidden)
        if i != input_len - 1:
            loss += criterion(output[0], input_variable[i + 1])

    if loss.data[0] != 0:
        loss.backward()
    encoder_optimizer.step()

# initializing the encoder
encoder = EncoderRNN(lang.nwords, hidden_size)
criterion = nn.NLLLoss()

encoder_optimizer = optim.SGD(encoder.parameters(), 0.001)

# training section begins
print("Training here...")
for i in range(200, 600):
    print(i)
    input_variable = variableFromSentence(lang, sentences[i])
    train(input_variable, encoder, encoder_optimizer, criterion)

# sampling the first 20 words from the seed (can be modified later)
def sample(starter, encoder):
    ret = []
    output = firstWord(lang, starter)
    hidden = encoder.initHidden()
    ret.append(starter)
    for i in range(20):
        output, hidden = encoder(output, hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0][0]
        word = lang.index2word[topi]
        ret.append(word)
        output = firstWord(lang, word)
    return ret

w = "the"
wformat = firstWord(lang, w)
sample(w, encoder)
