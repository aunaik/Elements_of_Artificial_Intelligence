# AdaBoost
## Description:
### Training images:
In this algorithm, we used 500 decision stumps to give the orientation of image.We are constructing 6 classifiers i.e.
0 vs 90,0 vs 180,0 vs 270,90 vs 180,90 vs 270 ,and 180 vs 270.Here,we used one vs one classifiers.Each decision stump
considers random pair of pixels.Each pair is generated  by using random.randint(0,191).Learner algorithm checks if
pixel a < pixel b of pair(a,b).Then it assigns the label to image.In order to find the most probable label for the
image, we checked all the images in examples.here we constructed examples by considering their given orientation.
For example, for 0 vs 90  classifiers we considered images whose orientation label had labeled as 0.
We found the label  which occurred maximum times in example data set for a< b condition and stored it in res. We also
computed the label which occurred maximum times in example data set for a> b condition and stored it in greater\_res.
if pixel a< pixel b then we assigned label which stored in res else we assigned the label stored in greater\_sum.
We computed labels for each image in examples.Then we assigned error and weight on the basis of their correct label.
We stored model in adaboost\_model.txt.We used this model for test data set.

### Test data:
We developed the model using adaboost\_model.txt.We computed the orientation of image by comparing the result of each
classifiers e.g 0 vs 90.We compared the values returned by each classifier, we computed the maximum occurrences of
result by adding weight of each classifier. The highest vote for orientation e.g north then we considered it as label
for that image.If tie occurs then we considered both orientations, we decided the final vote by running the classifier
which classify the images between these two orientations.We considered the result of this classifier as final result.

## Algorithm:
### Train:
We created examples of each classifier from training file specified in the command.We considered their orientation
given in test file. e.g.For classifier 0 vs 90, we created examples090 which stored all images who had orientation as
0 or 90.We did same for all classifier.We performed adaboost operation.We considered random 1000 pairs.
We initialized weight as 1/N where N is total number of examples.We assigned the label for each image as described
above.We calculated error for each image if image's label and its classified label are different.we used formula to
calculate error as error = error +weight[image].If classifier correctly assigned orientation of the image ,we need to
decrease the weight of the image.We used following formula :weight[image]= weight[image]\* error/(1-error)
we calculated the weight of each hypothesis by calculating log(1-error)/error.We created dictionary which stores pair
as key and another dictionary as value.This dictionary stores  2 keys.
1. key as '<' and its value as classified label when a < b in pair (a,b).
2. key as '>' and its value as classified label when a > b in pair (a,b).
We computed hypothesis weights as log (1-error/error).We saved model as orientation1orientation2 Pair < orientation\_
label weight > orientation\_label weight
e.g 090 (23,12) < 90 0.20 > 0 10 here it shows that for 0 vs 90 classifier ,pixel 23 and pixel 12 of image is compared
if pixel 23 < pixel 12 then orientation is 90 with weight 0.20  else orientation is 0.We stored this dictionary in
model file.
### Test:
We created orientation\_model dictionary using model file. For each test image,we computed result for each classifiers.
Then we find the maximum vote from these classifiers.If tie occurs then we check the vote of those two orientations ,
then we return the result.e.g.if 0 and 180 have same votes,we run 0 vs 180 classifier and assign its result as final
result.We calculated accuracy.
We had discussed adaboost with Surbhi Paithankar(spaithan)

# KNN:
## Design decision:
KNN is a lazy classifier, i.e. it never trains a model, so our model file simply writes the train file by removing the
train file labels. The testing part is very simple, we use euclidean distance formula to calculate distance of each test
image with all the training images and consider the nearest k images. Then voting is been done, the class with maximum
images in the k nearest images is been assigned to the test image under consideration.
Experimentation and problem faced:
The only parameter we had to tune here was k, we considered different value of k to check for better accuracy. We observed
that sqrt(n) where n is number of test images gives us the best accuracy. Finding out this value was actually very tedious.

# Neural Network
## Design decision:
We have considered one hidden layer(total 2 layers) fully connected feed forward neural network and sigmod function as
an activation function. For number of perceptrons, we tried different number of neurons and checked for accuracy. For learning
rate, we considered static learning rate (got by brute force) which gave us the best accuracy, although dynamically changing rate might give us
better accuracy. We have considered batch input instead of stochastic. The initial weights are randomly assigned between -0.5 to 0.5.
Experimentation and problem faced:
The biggest roadblock here was to choose parameters, we tried out different combination of parameters and found out the best we can get, although
there might be better parameters value giving better accuracy. The whole parameter selection process was very tedious.
Had discussion about how backpropogation works with Chetan.

# Result:

## For Adaboost:
       Random pairs       Accuracy        Running Time(approx)
          100              65.42%             3 mins
          500              68.29%             13 mins
          1000             70.51%             37 mins

## For KNN:
       value of k         Accuracy        Running time(approx)
            8              69.78%             4 mins
            5              69.98%             4 mins
           20              70.41%             4 mins
           30              70.41%             4 mins
           60              71.05%             4 mins
          192              71.16%             4 mins


## For Neural Network:
      epoch       hiddden neurons        Alpha        Accuracy        Running time(approx)
       1000             8                 0.28         69.57%             13 mins
       1000             8                 0.3          69.88%             13 mins
       1000             8                 0.33         69.88%             13 mins
       1000             8                 0.29         70.63%             13 mins
       1000            30                 0.29         69.88%             13 mins
      10000             8                 0.29         69.46%             25 mins
      10000            26                 0.29         70.09%             25 mins
       5000            26                 0.29         70.20%             18 mins
      10000            30                 0.29         71.47%             25 mins
      20000            30                 0.29         71.90%             40 mins

**Comparing the accuracies and the execution time for each classifier**,
If training happens offline and training time is not that important to the potential client then I'll recommend Neural Net with
learning rate 0.29, 1 hidden layer, 30 perceptron and 20000 epoch.
If the execution time (in particularly training time) is very important to the client and the training happens online then I'll
recommend them KNN with k=sqrt(n) where n is length of training data even though Neural Net gives better result, as it takes overall
around 4 mins to execute completely for the given data while neural net takes more time comparatively.

image test/8324860594.jpg not matched -- This is mostly because the light color in the image is at bottom and dark color as red in at the top.
image test/10484444553.jpg not matched -- Green color is at top and dark blue color is at the bottom.
image test/3714485886.jpg not matched -- Dark green color is at the top and light blue (road) is at the bottom which is not classified from train data.
image test/8259655038 orientation matches for all classifier
image test/82380849 orientation matches for all classifier
image test/17043724089 orientation matches for all classifier
General pattern is when an light or blue color concentration is in different are other that image top, classifier gives an incorrect orientation. Same is the case
when dark color appears in regions except ground area.

### Performance for varying train dataset, with epoch 1000 for nnet and k = 500 for adaboost
Train Dataset size        Nearest Accuracy          Adaboost Accuracy           NNet Accuracy
100                       63.3085896076%             60.7635206787%             53.9766702015%
500                       66.9141039236%             62.3541887593%             63.4146341463%
1000                      66.5959703075%             62.8844114528%             64.0509013786%
10000                     70.7317073171%             68.7168610817%             68.7168610817%
20000                     71.6861081654%             68.0805938494%             69.5652173913%

From the performance for varying datasets size, it can be inferred that the for small size of train data KNN gives better accuracy than Adaboost and Neural Net.
But with increasing input train dataset size accuracy for neural net increases with a greater factor that other 2 methods.

