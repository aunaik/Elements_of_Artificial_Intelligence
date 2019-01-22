# Part-of-Speech Tagging (POS)


## Design 

In this program, we computed emission probability,transition probability,initial probability and
probability of pos.We calculated the probability of pos e.g. probability of noun in training set by dividing
occurrences of each pos tag by total words in training set.Initial probability is stored in initial\_probability
dictionary.It stores tag as key and probability as its value. We calculated emission probability by dividing the
occurrences of a word in training set by occurrences of the tag. e.g. For calculating the emission probability of
dog being noun then we found occurrences of dog and occurrences of noun. We divided occurrences of dog and
occurrences of noun.We stored the emission probability in prob\_wi\_si.This is a dictionary of dictionaries.Outer key
denotes the si i.e. pos tag e.g. noun,adj whereas inner key denotes the word.Here P(word/pos) is stored as its
value. Transition probability was found by dividing the occurrences of transition from state s1 to s2 by the
occurrences of state s1. Transition probability is stored in prob\_si1\_si. We stored the probability of pos tag in
prob\_si.

## Algorithm:

1. Simplified : We need to calculate si= arg max P(Si=si|W).We used naive bayes law.We computed
the P(W/Si) i.e. emission probability and P(Si) i.e.probability of state.We multiplied P(W/Si) * P(Si) and ignored
the probability of word in denominator.
2. HMM\_VE:
In this case, we implemented forward-backward algorithm for
calculating the sequence.In this algorithm ,we computed the forward probability by using the equations as follow
alpha(POS)=P(POS)\*P(word/POS) for initial state and for i =2 to N
alpha(POSi)=summation of POSi-1 of P(word/POSi)\*P(POSi/POSi-1)\*alpha(POSi-1)
for backward algorithm ,equations are as follow
B(POSi)=1
for i=n-1 to i
B(POSi=sum(P(word/POSi)\*P(POSi-1/POSi)\*B(POSi-1)) for POSi-1
P(POSi/word1..wordn)=alpha(POS)\*B(POS)
If P(word/POS) is not present then we assigned the random probability as 0.00000001.
3. HMM MAP:
We used viterbi algorithm to compute the MAP.In this algorithm, we considered if transition probability of word
given POS is not present in dictionary then we assigned the probability as 1/total word present in the training set

## Problem Faced:
In order to fix the random probability if word is not present, I used various values.But I got the best accuracy with
0.00000001. So I chose 0.00000001 as random probability.

## Result :
Following results are obtained by running the code on bc.test file
So far scored 2000 sentences with 29442 words.
                 Words correct:     Sentences correct:
0. Ground truth: 100.00%              100.00%
1. Simplified:    93.92%               47.45%
2. HMM VE:        95.12%               54.50%
3. HMM MAP:       94.91%               53.60%
