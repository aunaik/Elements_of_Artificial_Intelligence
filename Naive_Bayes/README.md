# Tweet Classification:

Using Naive Bayes Law and bag of word assumption, we create a Classification model and train it using the training data and predict the city a tweet belongs to.

## Design Decisions:
We are using Multinomial Naive Bayes Classification wherein we create a multinomial distribution of words for
each city present in the training data and predict the city to which the tweets from the testing data belong to.
Firstly, we are pre-processing the training data as well as the test data by removing the speical characters, extra spaces
and carriage return usning regex. We have also removed some of the english stop words and converted all the textual data to lower case.
Secondly, we are considering all the tokens present in the training data after pre-processing to create a multinomial
distribution for each city. When we encounter a word not present in a city we punish(multiply by a pseudo count) that city
by a factor.
We use Naive bayes law: P(Posterior) = P(Likelihood)\*P(Prior) to find how likely it is that a tweet belongs to a particular city
City with the maximum P(Posterior) value for a tweet will be classified as the city that tweet belong to.
Here we are ignoring the denominator of the Bayes law because it will be constant for all cities.

## Exprerimentation:
Initially we tried using bernoulli smoothing so that we don't get a zero probability while calcuating the likelihood matrix(in our
case a dictionary) by considering a word vector of all the distinct words present in the training data which gave us an accuracy of
58.6%. In the quest of improving accuracy as well as reducing the overhead of calculating probabilities of all the words present in the
training data for each city, we decided to use a pseudo count to punish a city if it doesn't contain a word present in a tweet while
calculating the posterior probability. By experimenting with different constants we found out that we were getting maximum accuracy at
pseudo count value 10^-6. Thus we hardcoded this value. Although the accuracy of our model may go down as the size of the training data
increases. In such case we have to use some other pseudo count value which changes dynamically (eg: 1/total\_word\_count\_in\_training\_data)
this might keep our accuracy approximately constant.

