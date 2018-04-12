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
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

lang = Lang("English")

f = open("pride.txt", "r")
lines = []
for line in f:
    lang.addSentence(line)
    lines.append(line)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.i2h = nn.Linear(hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, input_size)
        self.o2o = nn.Linear(hidden_size, input_size)

        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, word, hidden):
        embedded = self.embedding(word).view(1, 1, -1)
        # print("embedded", embedded.size())
        # print("hidden", hidden.size())
        input_combined = torch.cat((hidden, embedded), 1)
        # print(input_combined.size())
        new_hidden = self.i2h(embedded + hidden)
        output = self.i2o(embedded + hidden)
        output = self.dropout(output)
        # output = self.o2o(new_hidden + embedded)
        # output = self.softmax(output)
        return output, new_hidden

    def initHidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    return result

def firstWord(lang, word):
    indexes = indexesFromSentence(lang, word)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    return result

def train(input_variable, encoder, criterion):
    # initialize the hidden states of the encoder
    hidden = encoder.initHidden()
    encoder.zero_grad()
    loss = Variable(torch.zeros(1))
    for i in range(len(input_variable)):
        output, hidden = encoder(
            input_variable[i], hidden)
        # print("HEY", output[0].size())
        if i != len(input_variable) - 1:
            loss += criterion(output[0], input_variable[i + 1])

    if loss.data[0] != 0:
        loss.backward()

# just figuring out the size of the hidden state and stuff
hidden_size = 128
# initializing the encoder and reversecoder
encoder = EncoderRNN(lang.n_words, hidden_size)
criterion = nn.NLLLoss()

learning_rate = 0.0005
# the actual training part happens here (run for some epochs over the training data)
print("Training here...")
for i in range(200):
    print(i)
    input_variable = variableFromSentence(lang, lines[i])
    train(input_variable, encoder, criterion)

def sample(input_variable, encoder):
    encoder_hidden = encoder.initHidden()

    for i in range(100):
        hidden = encoder.initHidden()
        # encoder.zero_grad()
        loss = Variable(torch.zeros(1))
        for i in range(len(input_variable)):
            output, hidden = encoder(
                input_variable[i], hidden)
            print("OUTPUT", output)
            best_index = 0
            best_value = -1 * float("inf")
            for i in range(len(output.data)):
                if output.data[i] > best_value:
                    best_index = i
                    best_value = output.data[i]
            print("BEST INDEX", best_index)
            # topv, topi = output.data.topk(1)
            # # print("topi", topi)
            # topi = topi[0][0]
            # print(topi[0])
            # word = lang.index2word[topi[0]]

        # encoder_output, encoder_hidden = encoder(encoder_output, encoder_hidden)
        # print("OUTPUT", encoder_output)
        # input = encoder_output
        # print(encoder_output[0].size(), "SIZE")
        # print("HIHIHI", encoder_output[0].sum())
        # topv, topi = encoder_output.data.topk(1)
        # print("topi", topi)
        # # topi = topi[0][0]
        #
        # word = lang.index2word[topi]
        # print("WORD", word)

w = "The"
wformat = firstWord(lang, w)
# inputt = Variable(torch.LongTensor([[23]]))
# print(wformat.size(), "HEEYYY")
sample(wformat, encoder)
