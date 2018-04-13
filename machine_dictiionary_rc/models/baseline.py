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

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        # self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        word.lower()
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

lang = Lang("English")

# f = open("pride.txt", "r")
# lines = []
# for line in f:
#     lang.addSentence(line)
#     lines.append(line)

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
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, word, hidden):
        embedded = self.embedding(word).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        new_hidden = self.i2h(embedded + hidden)
        output = self.i2o(embedded + hidden)
        output = self.dropout(output)
        # output = self.softmax(output)

        hidden = new_hidden
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

# just figuring out the size of the hidden state and stuff
hidden_size = 256
# initializing the encoder and reversecoder
encoder = EncoderRNN(lang.n_words, hidden_size)
criterion = nn.NLLLoss()

encoder_optimizer = optim.SGD(encoder.parameters(), 0.001)

# the actual training part happens here (run for some epochs over the training data)
print("Training here...")
for i in range(200, 400):
    print(i)
    input_variable = variableFromSentence(lang, sentences[i])
    # print(lines[i])
    train(input_variable, encoder, encoder_optimizer, criterion)

def sample(starter, encoder):
    output = firstWord(lang, starter)
    hidden = encoder.initHidden()
    print(starter)
    for i in range(20):
        output, hidden = encoder(
            output, hidden)
        topv, topi = output.data.topk(1)
        topi = topi[0][0]
        word = lang.index2word[topi[0]]
        print(word)
        output = firstWord(lang, word)

w = "the"
wformat = firstWord(lang, w)
sample(w, encoder)
