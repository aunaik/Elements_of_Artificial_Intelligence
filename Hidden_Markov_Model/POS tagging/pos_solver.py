###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids:
# Akshay Naik(aunaik),Ameya Angal(aangal),Praneta Paithankar(ppaithan)
# (Based on skeleton code by D. Crandall)
#
# Design:
# In this program, we computed emission probability,transition probability,initial probability and
# probability of pos.We calculated the probability of pos e.g. probability of noun in training set by dividing
# occurrences of each pos tag by total words in training set.Initial probability is stored in initial_probability
# dictionary.It stores tag as key and probability as its value. We calculated emission probability by dividing the
# occurrences of a word in training set by occurrences of the tag. e.g. For calculating the emission probability of
# dog being noun then we found occurrences of dog and occurrences of noun. We divided occurrences of dog and
# occurrences of noun.We stored the emission probability in prob_wi_si.This is a dictionary of dictionaries.Outer key
#  denotes the si i.e. pos tag e.g. noun,adj whereas inner key denotes the word.Here P(word/pos) is stored as its
# value. Transition probability was found by dividing the occurrences of transition from state s1 to s2 by the
# occurrences of state s1. Transition probability is stored in prob_si1_si. We stored the probability of pos tag in
# prob_si.
# Algorithm:
# 1.Simplified : We need to calculate si= arg max P(Si=si|W).We used naive bayes law.We computed
# the P(W/Si) i.e. emission probability and P(Si) i.e.probability of state.We multiplied P(W/Si) * P(Si) and ignored
# the probability of word in denominator.
# 2.HMM_VE:
# In this case, we implemented forward-backward algorithm for
# calculating the sequence.In this algorithm ,we computed the forward probability by using the equations as follow
# alpha(POS)=P(POS)*P(word/POS) for initial state and
# for i =2 to N
# alpha(POSi)=summation of POSi-1 of P(word/POSi)*P(POSi/POSi-1)*alpha(POSi-1)
# for backward algorithm ,equations are as follow
# B(POSi)=1
# for i=n-1 to i
# B(POSi=sum(P(word/POSi)*P(POSi-1/POSi)*B(POSi-1)) for POSi-1
# P(POSi/word1..wordn)=alpha(POS)*B(POS)
# If P(word/POS) is not present then we assigned the random probability as 0.00000001.
# 3.HMM MAP:
# We used viterbi algorithm to compute the MAP.In this algorithm, we considered if transition probability of word
# given POS is not present in dictionary then we assigned the probability as 1/total word present in the training set
# Problem Faced:
# In order to fix the random probability if word is not present, I used various values.But I got the best accuracy with
# 0.00000001.So I chose 0.00000001 as random probability.
# Result :
# Following results are obtained by running the code on bc.test file
# So far scored 2000 sentences with 29442 words.
#                  Words correct:     Sentences correct:
# 0. Ground truth: 100.00%              100.00%
# 1. Simplified:    93.92%               47.45%
# 2. HMM VE:        95.12%               54.50%
# 3. HMM MAP:       94.91%               53.60%
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
