# Optical Character Recognition (OCR)

The goal here is to recognize text in an Image (Character by character).
We consider a simplied OCR problem in which the font and font size is known ahead of time,
but the basic technique we'll use is very similar to that used by commercial OCR systems

## Problem Formulation and description:

We are using three different methods (Simplified Bayes Net, Variable elimination on HMM and Viterbi algorith on HMM  )
to recognize the text in image, which will help us to understand which performs better than the others.
We consider HMM as the possibility of second character being something depends on the recognized first character as well as the
remaining observed characters and so on, thus it forms a HMM of characters where observed values are the image we see (test image).
Firstly, we are using bc.train as a texual training data file to calcualte the initial probabilities as well as
the transition probabilities. This file contains words with their POS tag which we have ignored. We are calculating emission
probabilities by comparing the obesrved character's pixels data with the pixels information of each of the character in the
image training data.
After we have initial probabilities, transition probabilities and emission probabilities we implement differet methods to
recognize the text in an image provided as test-image file and display the recognized text.
Secondly, for calculating emission probabilities we have tried to take advantage of the pixel information we have. While calculating
emission probabilities we consider two different scenarios, first when the test image have densely populated black pixels and second
when the image have sparse black pixels as compared to the training image pixals.
For both the scenarios we assigned different weights to matching black pixals, matching white pixals, non matching pixals where the test
image character's pixal is white and train image character's pixal is black  and vice versa.
By doing this, we were able to see a drastic improvement in the efficiency of image text recognition.


## Experimentation and problems faced:

Firstly, to calculate the emission probabilities (we haven't used prior probabilities as it's always constant), we tried many ways like by just Calculating the number of pixels matches and then divide
that with the total number of pixals, using Naive Bayes where we gave 0.8 probability value when the pixals matches and 0.2 probability
value when the pixals doesn't matches. At the end we came up with using different probability weights for different pixal values by discussing
with Ninaad Joshi. Credit goes to Ninaad who brought up the fact that density of the image will also be a factor in deciding
the weights we assign to white matching pixals, black matching pixals and non-matching pixals.
Secondly, when we don't have a particular character from the test image in either our initial probability table or transition probability table,
we have assigned a constant probability. We tried differet constant probability values for V.E as well as Viterbi and set the one for which
we got maximum accuracy.
Thirdly, we faced a lot of issues in variable elimination as after a point of time the probability values were underflowing, in order to take care
of this we divided the probability values in each iteration of forward and backward algorithm by maximum probability value. Prof. Crandall gave us this
idea to scale the probability values in each iteration.
