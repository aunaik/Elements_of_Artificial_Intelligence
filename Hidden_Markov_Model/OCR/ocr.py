#!/usr/bin/env python
# B551 Elements of AI, Prof. David Crandall
# Author: Akshay Naik
# (based on skeleton code by D. Crandall, Oct 2017)


import copy
import heapq
import math
import os
import re
import sys

from PIL import Image

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25


class Ocr:
    def __init__(self):
        self.init_prob = {}
        self.trans_prob = {}
        self.emission_prob = {}

    # Load letters
    def load_letters(self, fname):
        im = Image.open(fname)
        px = im.load()
        (x_size, y_size) = im.size
        result = []
        for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
            result += [['*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg + CHARACTER_WIDTH) for y in range(0, CHARACTER_HEIGHT)], ]
        return result

    # Load training letters
    def load_training_letters(self, fname):
        TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
        letter_images = self.load_letters(fname)
        return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}

    # Calculate the total number of letters in the dictionary which is sent as argument
    def total_letter_count(self, letter_list):
        return sum(letter_list[letter] for letter in letter_list)

    # Calculate the probability value for the given dictionary from the frequency list
    def cal_prob(self):
        for char in self.trans_prob:
            sum_dep_freq = self.total_letter_count(self.trans_prob[char])
            for letter in self.trans_prob[char]:
                self.trans_prob[char][letter] = self.trans_prob[char][letter] / float(sum_dep_freq)

    # Calculate initial probability and transition probability
    def calculate_trans_prob_init_prob(self, train_txt_fname):
        # d = os.path.dirname(__file__)
        f = open(train_txt_fname, 'r');
        line_number = 0
        for line in f:
            line_number += 1
            # We have used bc.train so we are considering only the even positioned words of the line, for different text training file we can use the below commented line.
            # exemplars = list(re.sub(r'[&|$|*|;|`|#|@|%|^|~|/|<|>|:|[|\]|{|}|+|=|_]', r'', " ".join([w for w in line.split()])))
            exemplars = list(re.sub(r'[&|$|*|;|`|#|@|%|^|~|/|<|>|:|[|\]|{|}|+|=|_]', r'', " ".join([w for w in line.split()][0::2])))
            if exemplars:
                self.init_prob[exemplars[0]] = self.init_prob.get(exemplars[0], 0) + 1
                for letter in range(1, len(exemplars)):
                    if exemplars[letter - 1] in self.trans_prob:
                        self.trans_prob[exemplars[letter - 1]][exemplars[letter]] = self.trans_prob[exemplars[letter - 1]].get(exemplars[letter], 0) + 1
                    else:
                        temp = {exemplars[letter]: 1}
                        self.trans_prob[exemplars[letter - 1]] = temp
        sum_init = self.total_letter_count(self.init_prob)
        for x in self.init_prob:
            self.init_prob[x] = float(self.init_prob[x]) / sum_init
        self.cal_prob()

    # Calculate the emission probability
    def calculate_emission_prob(self):
        blackCountInTestLetter = 0
        blackCountInTrainLetter = 0
        for letter in test_letters:
            for i in letter:
                if i == '*':
                    blackCountInTestLetter += 1
        for letter in train_letters:
            for i in train_letters[letter]:
                if i == '*':
                    blackCountInTrainLetter += 1
        for test_letter in range(len(test_letters)):
            self.emission_prob[test_letter] = {}
            for train_letter in train_letters:
                black_count = 0
                white_count = 0
                black_non_matching = 0
                white_non_matching = 0
                test_black_density = 0
                train_black_density = 0
                total = 0
                for char in range(len(test_letters[test_letter])):
                    total += 1
                    if test_letters[test_letter][char] == train_letters[train_letter][char] and \
                                    train_letters[train_letter][char] == '*':
                        black_count += 1
                    elif test_letters[test_letter][char] == train_letters[train_letter][char] and \
                                    train_letters[train_letter][char] == ' ':
                        white_count += 1
                    elif train_letters[train_letter][char] == '*':
                        black_non_matching += 1
                    elif train_letters[train_letter][char] == ' ':
                        white_non_matching += 1

                if blackCountInTestLetter / len(test_letters) > blackCountInTrainLetter / len(train_letters):
                    self.emission_prob[test_letter][train_letter] = math.pow(0.8, black_count) * math.pow(0.7,white_count) * math.pow(0.3, black_non_matching) * \
                                                                    math.pow(0.2, white_non_matching)
                else:
                    self.emission_prob[test_letter][train_letter] = math.pow(0.99, black_count) * math.pow(0.7,white_count) * math.pow(0.3, black_non_matching) * \
                                                                    math.pow(0.01, white_non_matching)

    # We have used simplified Bayes net here to recognize the text in the image, where we estimate the most probable character the image has
    # at a particular position based on the emission probability distribution of the observed character.
    def simplified(self):
        simple = ""
        for char in self.emission_prob:
            simple += "".join(max(self.emission_prob[char], key=lambda x: self.emission_prob[char][x]))
        print " Simple: {0}".format(simple)

    # Calculate probability using forward algorithm
    def forward_algorithm(self):
        fwd = []
        f_prev = {}
        for i, word in enumerate(test_letters):
            f_curr = {}
            for j, type in enumerate(train_letters):
                if i == 0:
                    prev_sum = self.init_prob.get(type, math.pow(10, -6))
                else:
                    prev_sum = sum(
                        f_prev[k] * self.trans_prob.get(k, {}).get(type, math.pow(10, -6)) for k in train_letters)
                f_curr[type] = prev_sum * self.emission_prob[i].get(type, math.pow(10, -6))
            max_no = max(f_curr.values())
            f_curr = {key: value / max_no for key, value in f_curr.iteritems()}
            fwd.append(f_curr)
            f_prev = f_curr
        return fwd

    # Calculate probability using backward algorithm
    def backward_algorithm(self):
        bkw = []
        b_prev = {}
        next_list = test_letters[::-1]
        for i, word in enumerate(next_list):
            b_curr = {}
            for j, type in enumerate(train_letters):
                if i == 0:
                    b_curr[type] = 1
                else:
                    next_word = i - 1
                    b_curr[type] = sum(
                        b_prev[next_state] * self.emission_prob[next_word].get(next_state, math.pow(10, -6)) *
                        self.trans_prob.get(type, {}).get(next_state, math.pow(10, -6)) for k, next_state in
                        enumerate(train_letters))
            max_no = max(b_curr.values())
            b_curr = {key: value / max_no for key, value in b_curr.iteritems()}
            bkw.append(b_curr)
            b_prev = b_curr
        return bkw

    # We have used forward-backward algorithm here to perform variable elimination on HMM of the characters present in the image(test image).
    # By considering all the observed characters from the test image, we estimate the most likely character at a particular position by marginalizing
    # rest of the characters around.
    def hmm_ve(self):
        # forward algorithm
        fwd=self.forward_algorithm()
        # backward_algorithm
        bkw=self.backward_algorithm()
        # find posterior sequence max for each state
        sequence = []
        k = len(test_letters) - 1
        for i in range(len(test_letters)):
            prob_dict = {type: bkw[k][type] * fwd[i][type] for type in train_letters}
            sequence.append(max(prob_dict, key=prob_dict.get))
            k -= 1
        print " HMM VE: {0}".format(''.join(sequence))

    # Here we use viterbi algorithm on HMM of the characters present in the image to estimate the most probable sequence of the characters present
    # in the test image (maximum a posteriori (MAP)).
    def hmm_map(self):
        currentState = [None] * 128
        previousState = [None] * 128
        for stateIndex, state in enumerate(test_letters):
            for index, currentletter in enumerate(train_letters):
                if stateIndex == 0:
                    result = -math.log(self.emission_prob[0][currentletter]) - math.log(
                        self.init_prob.get(currentletter, math.pow(10, -8)))
                    currentState[ord(currentletter)] = [result, [currentletter]]
                else:
                    meanheap = []
                    for indexForPreviousLetter, previousLetter in enumerate(train_letters):
                        transProbAndPreviousProbablity = -math.log(
                            self.trans_prob.get(previousLetter, {}).get(currentletter, math.pow(10, -8))) + \
                                                         previousState[ord(previousLetter)][0]
                        meanheap.append([transProbAndPreviousProbablity, previousState[ord(previousLetter)][1] + [currentletter]])
                    heapq.heapify(meanheap)
                    maxFromPreviousAndTrransition = heapq.heappop(meanheap)
                    result = maxFromPreviousAndTrransition[0] - math.log(self.emission_prob[stateIndex][currentletter])
                    currentState[ord(currentletter)] = [result, maxFromPreviousAndTrransition[1]]
            previousState = copy.deepcopy(currentState)
            currentState = [None] * 128
        final = []
        for element in previousState:
            if element is not None:
                final.append(element)
        heapq.heapify(final)
        result = heapq.heappop(final)
        print "HMM MAP: {0}".format(''.join(result[1]))

# main program
ocr = Ocr()
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = ocr.load_training_letters(train_img_fname)
test_letters = ocr.load_letters(test_img_fname)
ocr.calculate_trans_prob_init_prob(train_txt_fname)
ocr.calculate_emission_prob()
ocr.simplified()
ocr.hmm_ve()
ocr.hmm_map()
