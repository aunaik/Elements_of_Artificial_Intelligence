###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids:
# Akshay Naik(aunaik),Ameya Angal(aangal),Praneta Paithankar(ppaithan)
# (Based on skeleton code by D. Crandall)

# References:
# http://www.cs.cmu.edu/~guestrin/Class/10701/slides/hmms-structurelearn.pdf
# https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
# we have discussed the working of variable elimination algorithm with Aishwarya Dhage

import copy
import heapq
import math

pos_type = ['noun', 'adj', 'adv', 'adp', 'conj', 'det', 'num', 'pron', 'prt', 'verb', 'x', '.']


class Solver:
    def __init__(self):
        self.frequency_dict = {o: {} for o in pos_type}
        self.total_pos = {o: 0.0 for o in pos_type}
        self.total_word = 0
        self.prob_si = {}
        self.prob_wi_si = {o: {} for o in pos_type}
        self.freq_si1_si = {o: {key: 0 for key in pos_type} for o in pos_type}
        self.prob_si1_si = {o: {key: 0 for key in pos_type} for o in pos_type}
        self.initial_probability = {o: 0.0 for o in pos_type}

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        prob = 1.0
        random_prob = 0.00000001
        for index, word in enumerate(sentence):
            if index == 0:
                prob *= self.prob_wi_si[label[index]].get(word, random_prob) * self.initial_probability[label[index]]
            else:
                prob *= self.prob_wi_si[label[index]].get(word, random_prob) * self.prob_si1_si[label[index - 1]].get(
                    label[index], random_prob)
        return math.log(prob)

    # Do the training!
    def train(self, data):
        for item in data:
            self.initial_probability[item[1][0]] += 1
            for i in range(len(item[0])):
                self.total_pos[item[1][i]] += 1
                self.total_word += 1
                if item[0][i] in self.frequency_dict[item[1][i]]:
                    self.frequency_dict[item[1][i]][item[0][i]] += 1
                else:
                    self.frequency_dict[item[1][i]][item[0][i]] = 1
                if i != len(item[0]) - 1:
                    self.freq_si1_si[item[1][i]][item[1][i + 1]] += 1
        self.prob_si = {key: self.total_pos[key] / self.total_word for key in pos_type}
        for key in pos_type:
            for elements, value in self.frequency_dict[key].iteritems():
                self.prob_wi_si[key][elements] = value / self.total_pos[key]
            for k1, val1 in self.freq_si1_si[key].iteritems():
                prob = val1 / self.total_pos[key]
                self.prob_si1_si[key][k1] = prob if prob != 0 else 0.00000001
        length = len(data)
        for key, value in self.initial_probability.iteritems():
            self.initial_probability[key] = value / length

    # Functions for each algorithm.
    # Find sequence by using simplified algorithm
    def simplified(self, sentence):
        sequence = []
        for word in sentence:
            prob = 0
            word_type = ""
            for type in pos_type:
                if word in self.prob_wi_si[type]:
                    current = self.prob_wi_si[type][word] * self.prob_si[type]
                    if current > prob:
                        prob = current
                        word_type = type
                else:
                    current = 0.0000001
                    if current > prob:
                        prob = current
                        word_type = type

            sequence.append(word_type)
        return sequence

    # calculate probability by using forward algorithm
    def forward_algorithm(self,sentence):
        rand_prob = 0.00000001
        fwd = []
        f_prev = {}
        for i, word in enumerate(sentence):
            f_curr = {}
            for type in pos_type:
                if i == 0:
                    prev_sum = self.initial_probability[type]
                else:
                    prev_sum = sum(f_prev[k] * self.prob_si1_si[k][type] for k in pos_type)
                f_curr[type] = prev_sum * self.prob_wi_si[type].get(word, rand_prob)
            fwd.append(f_curr)
            f_prev = f_curr
        return fwd

    # calculate probability by using backward algorithm
    def backward_algorithm(self,sentence):
        rand_prob = 0.00000001
        bkw = []
        b_prev = {}
        next_list = sentence[::-1]
        for i, word in enumerate(next_list):
            b_curr = {}
            for type in pos_type:
                if i == 0:
                    b_curr[type] = 1
                else:
                    next_word = next_list[i - 1]
                    b_curr[type] = sum(
                        b_prev[k] * self.prob_wi_si[k].get(next_word, rand_prob) * self.prob_si1_si[type][k] for k in
                        pos_type)
            bkw.append(b_curr)
            b_prev = b_curr
        return bkw

    # calculate the sequence by using hmm variable elimination
    def hmm_ve(self, sentence):
        fwd=self.forward_algorithm(sentence)
        bkw=self.backward_algorithm(sentence)
        # find posterior sequence max for each state
        sequence = []
        k = len(sentence) - 1
        for i in range(len(sentence)):
            prob_dict = {type: bkw[k][type] * fwd[i][type] for type in pos_type}
            sequence.append(max(prob_dict, key=prob_dict.get))
            k -= 1
        return sequence

    # calculate the sequence by using viterbi algorithm
    def hmm_viterbi(self, sentence):
        currentState = {}
        previousState = {}
        for stateIndex, state in enumerate(sentence):
            for currentWord in pos_type:
                if stateIndex == 0:
                    # for currentWord in pos_type:
                    emission = -math.log(self.prob_wi_si[currentWord].get(state, (1.0 / self.total_word)))
                    initial = -math.log(self.initial_probability.get(currentWord, (1.0 / self.total_word)))
                    result1 = emission + initial
                    currentState[currentWord] = [result1, [currentWord]]
                else:
                    # for currentWord in pos_type:
                    meanheap = []
                    for previousWord in pos_type:
                        transitionProb = -math.log(self.prob_si1_si[previousWord].get(currentWord, math.pow(10, -6)))
                        previousResult = previousState[previousWord][0]
                        transProbAndPreviousProbablity = transitionProb + previousResult
                        meanheap.append(
                            [transProbAndPreviousProbablity, previousState[previousWord][1] + [currentWord]])
                    heapq.heapify(meanheap)
                    maxFromPreviousAndTrransition = heapq.heappop(meanheap)
                    emissionProb = - math.log(self.prob_wi_si[currentWord].get(state, 1.0 / self.total_word))
                    result = maxFromPreviousAndTrransition[0] + emissionProb
                    currentState[currentWord] = [result, maxFromPreviousAndTrransition[1]]
            previousState.clear()
            previousState = copy.deepcopy(currentState)
            currentState.clear()
        finalresult = min(previousState.values(), key=lambda x: x[0])[1]
        return finalresult

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print "Unknown algo!"
