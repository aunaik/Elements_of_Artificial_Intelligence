#!/usr/bin/python2

import random
import math
import sys
import numpy as np
import operator
from collections import Counter

class trainImage:
    def __init__(self):
        self.imageName = ''
        self.orientationLabel = None
        self.features = []

# Activation function for each perceptron in neural network
def sigmoidvalue(mat):
    mat = np.clip(mat, -500, 500)
    return 1.0 / (1 + np.exp(-mat))


# adaboost alogrithm for training data
def adaboost(examples, orientation1, orientation2):
    N = len(examples)
    w = {}
    z = {}
    h = {}
    for example in examples:
        w[example] = float(1) / N
    i = 0
    for k in range(1000):
        total_weight = 0
        no1 = random.randint(0, 191)
        no2 = random.randint(0, 191)
        currentpair = no1, no2
        d = {}
        correct_orientation = {}
        # learner code
        learner_dict = {orientation1: 0, orientation2: 0}
        learner_dict_greater = {orientation1: 0, orientation2: 0}
        for image in examples:
            if image.features[no1] < image.features[no2]:
                learner_dict[image.orientationLabel] += 1
            else:
                learner_dict_greater[image.orientationLabel] += 1
        res = max(learner_dict, key=learner_dict.get)
        greater_res = max(learner_dict_greater, key=learner_dict_greater.get)
        # label assignment
        for example in examples:
            if example.features[no1] < example.features[no2]:
                d[example] = res
            else:
                d[example] = greater_res
            correct_orientation[example] = example.orientationLabel
        h[currentpair] = d
        error = 0
        # error calculation and weight calculation
        for key in h[currentpair]:
            if h[currentpair][key] != correct_orientation[key]:
                error = error + w[key]
        for key in h[currentpair]:
            if h[currentpair][key] == correct_orientation[key]:
                w[key] = error * w[key] / (1 - error)
            total_weight += w[key]
        # normalize
        for key in h[currentpair]:
            w[key] /= total_weight
        if error != 0:
            z[currentpair] = {'<': [res, math.log(abs(1 - error) / error)],
                              '>': [greater_res, math.log(abs(1 - error) / error)]}
        else:
            z[currentpair] = {'<': [res, 1],
                              '>': [greater_res, 1]}
        i += 2
    return z


# test the data using adaboost model
def testAdaboost(orientation_model):
    image_orientation = {}
    with open(sys.argv[2], 'r')as f:
        images = []
        for line in f.readlines():
            image = trainImage()
            imageDetails = line.split(" ")
            image.imageName = imageDetails[0]
            image.orientationLabel = int(imageDetails[1])
            image.features = map(float, imageDetails[2:])
            images.append(image)
    for image in images:
        d = {0: 0, 90: 0, 180: 0, 270: 0}
        res = assign_orientation(image, 0, 90, orientation_model)
        d[res] += 1
        res = assign_orientation(image, 0, 180, orientation_model)
        d[res] += 1
        res = assign_orientation(image, 0, 270, orientation_model)
        d[res] += 1
        res = assign_orientation(image, 90, 180, orientation_model)
        d[res] += 1
        res = assign_orientation(image, 90, 270, orientation_model)
        d[res] += 1
        res = assign_orientation(image, 180, 270, orientation_model)
        d[res] += 1

        res = max(d.iteritems(), key=operator.itemgetter(1))
        max_value = [k for k, v in d.items() if v == res[1]]
        if len(max_value) == 2:
            d1 = {max_value[0]: 0, max_value[1]: 0}
            if max_value[0] > max_value[1]:
                max_value[0], max_value[1] = max_value[1], max_value[0]
            dict_key = "{0}{1}".format(max_value[0], max_value[1])
            for key in orientation_model[dict_key]:
                indexes = key.replace("(", "").replace(")", "").split(",")
                i1 = int(indexes[0])
                i2 = int(indexes[1])
                if image.features[i1] < image.features[i2]:
                    o = orientation_model[dict_key][key]["<"][0]
                    d1[o] += orientation_model[dict_key][key]["<"][1]
                else:
                    o = orientation_model[dict_key][key][">"][0]
                    d1[o] += orientation_model[dict_key][key][">"][1]
            res = max(d1, key=d1.get)
            image_orientation[image.imageName] = [image.orientationLabel, res]
        else:
            image_orientation[image.imageName] = [image.orientationLabel, res[0]]
    return image_orientation


# assign orientation to test image
def assign_orientation(image, o1, o2, orientation_model):
    dict_key = "{0}{1}".format(o1, o2)
    d = {o1: 0, o2: 0}
    for key in orientation_model[dict_key]:
        indexes = key.replace("(", "").replace(")", "").split(",")
        i1 = int(indexes[0])
        i2 = int(indexes[1])
        if image.features[i1] < image.features[i2]:
            o = orientation_model[dict_key][key]["<"][0]
            d[o] += orientation_model[dict_key][key]["<"][1]
        else:
            o = orientation_model[dict_key][key][">"][0]
            d[o] += orientation_model[dict_key][key][">"][1]
    res = max(d, key=d.get)
    return res


# train ada boost
def train_adaboost():
    with open(sys.argv[2], 'r') as f:
        examples090 = []
        examples90180 = []
        examples180270 = []
        examples0180 = []
        examples0270 = []
        examples90270 = []
        for line in f.readlines():
            image = trainImage()
            image_details = line.split(" ")
            image.imageName,image.orientationLabel,image.features= image_details[0],int(image_details[1]),map(float, image_details[2:])
            if image.orientationLabel == 0:
                examples090.append(image)
                examples0180.append(image)
                examples0270.append(image)
            if image.orientationLabel == 90 :
                examples90180.append(image)
                examples090.append(image)
                examples90270.append(image)
            if image.orientationLabel == 180 :
                examples180270.append(image)
                examples0180.append(image)
                examples90180.append(image)
            if image.orientationLabel == 270 :
                examples0270.append(image)
                examples180270.append(image)
                examples90270.append(image)

    z1 = adaboost(examples090, 0, 90)
    z2 = adaboost(examples0180, 0, 180)
    z3 = adaboost(examples0270, 0, 270)
    z4 = adaboost(examples90180, 90, 180)
    z5 = adaboost(examples90270, 90, 270)
    z6 = adaboost(examples180270, 180, 270)

    with open(sys.argv[3], 'w') as f:
        f.write("090")
        for key in z1:
            f.write("\t{0}".format(key))
            for k in z1[key]:
                f.write("\t{0}\t{1}\t{2}".format(k, z1[key][k][0], z1[key][k][1]))
        f.write("\n0180")
        for key in z2:
            f.write("\t{0}".format(key))
            for k in z2[key]:
                f.write("\t{0}\t{1}\t{2}".format(k, z2[key][k][0], z2[key][k][1]))
        f.write("\n0270")
        for key in z3:
            f.write("\t{0}".format(key))
            for k in z3[key]:
                f.write("\t{0}\t{1}\t{2}".format(k, z3[key][k][0], z3[key][k][1]))
        f.write("\n90180")
        for key in z4:
            f.write("\t{0}".format(key))
            for k in z4[key]:
                f.write("\t{0}\t{1}\t{2}".format(k, z4[key][k][0], z4[key][k][1]))
        f.write("\n90270")
        for key in z5:
            f.write("\t{0}".format(key))
            for k in z5[key]:
                f.write("\t{0}\t{1}\t{2}".format(k, z5[key][k][0], z5[key][k][1]))
        f.write("\n180270")
        for key in z6:
            f.write("\t{0}".format(key))
            for k in z6[key]:
                f.write("\t{0}\t{1}\t{2}".format(k, z6[key][k][0], z6[key][k][1]))


# Training KNN model
def train_nearest():
    file = open(sys.argv[3], "w")
    with open(sys.argv[2]) as inputfile:
        for line in inputfile:
            file.write(" ".join(line.split(" ")[1:]))

# Training Neural network
def train_nnet():
    # Inputting train data
    x = np.loadtxt(sys.argv[2],usecols=range(2, 194))
    # Inputting train labels
    label = np.loadtxt(sys.argv[2],dtype=np.int, usecols=(1,))

    d = {0: np.array([1, 0, 0, 0]),
         90: np.array([0, 1, 0, 0]),
         180: np.array([0, 0, 1, 0]),
         270: np.array([0, 0, 0, 1])}
    # True output
    y = map(lambda x: d[x], label)

    # Initializing random weights
    np.random.seed(1)
    w1 = (np.random.uniform(-0.5, 0.5, size=(30, 192)))
    b1 = (np.random.uniform(-0.5, 0.5, size=30))
    w2 = (np.random.uniform(-0.5, 0.5, size=(4, 30)))
    b2 = (np.random.uniform(-0.5, 0.5, size=4))
    alpha = 0.29
    n = len(x)

    for epoch in range(20000):
        # Forward propogation
        z1 = np.dot(x, w1.T) + b1
        a1 = sigmoidvalue(z1)
        z2 = np.dot(a1, w2.T) + b2
        a2 = sigmoidvalue(z2)

        # Back propogation
        delta2 = (a2 - y) * a2 * (1 - a2)
        delta1 = np.dot(delta2, w2) * a1 * (1 - a1)

        dw2 = np.dot(delta2.T, a1)
        dw1 = np.dot(delta1.T, x)

        # Updating weights
        w1 = w1 - (alpha / (n)) * dw1
        w2 = w2 - (alpha / (n)) * dw2
        b1 = b1 - (alpha / (n)) * np.sum(delta1, axis=0)
        b2 = b2 - (alpha / (n)) * np.sum(delta2, axis=0)

    file = open(sys.argv[3], "w")
    file.write("w1\n")
    np.savetxt(file, w1, delimiter=" ", fmt="%f")
    file.write("b1\n")
    np.savetxt(file, b1, delimiter=" ", fmt="%f")
    file.write("w2\n")
    np.savetxt(file, w2, delimiter=" ", fmt="%f")
    file.write("b2\n")
    np.savetxt(file, b2, delimiter=" ", fmt="%f")
    file.close()


# test ada boost
def test_adaboost():
    orientation_model = {}
    matching = 0
    with open(sys.argv[3], 'r') as f:
        for line in f.readlines():
            details = line.split("\t")
            d = {}
            for i in range(1, len(details), 7):
                d[details[i]] = {details[i + 1]: [int(details[i + 2]), float(details[i + 3])],
                                 details[i + 4]: [int(details[i + 5]), float(details[i + 6])]}
            orientation_model[details[0]] = d
    image_orientation = testAdaboost(orientation_model)
    total_test_images = 0
    with open("output.txt", 'w') as f:
        for image in image_orientation:
            total_test_images += 1
            #print 'img {0} is {1} and classified is {2}'.format(image, image_orientation[image][0],
                                                                #image_orientation[image][1])
            f.write("{0} {1}\n".format(image, image_orientation[image][1]))
            if image_orientation[image][0] == image_orientation[image][1]:
                matching += 1
    accuracy = float(matching) / total_test_images
    print 'Accuracy is {0}'.format(accuracy * 100)

# Test KNN
def test_nearest():
    correct_predict = 0
    train = []
    test = []

    with open(sys.argv[3]) as inputfile:
        for line in inputfile:
            train.append([line.split(" ")[0], np.array(line.split(" ")[1:], dtype=int)])

    k = int(math.sqrt(len(train)))

    with open(sys.argv[2]) as inputfile:
        for line in inputfile:
            test.append([line.split(" ")[0:2], np.array(line.split(" ")[2:], dtype=int)])

    file = open("output.txt", "w")
    for img_test in test:
        dist_scores = []
        # Calculating euclidean distance
        for img_train in train:
            e_dist = np.sum((img_train[1] - img_test[1]) ** 2)
            dist_scores.append([img_train[0], math.sqrt(e_dist)])
        knn = Counter([label[0] for label in sorted(dist_scores, key=lambda x: x[1])[0:k]])
        predicted_label = max(knn, key=lambda x: knn[x])
        file.write(" ".join([img_test[0][0], predicted_label, "\n"]))
        if img_test[0][1] == predicted_label:
            correct_predict += 1

    print "Accuracy = ", float(correct_predict) * 100 / len(test)


def test_nnet():
    test = np.loadtxt(sys.argv[2],usecols=range(2, 194))
    test_label = np.genfromtxt(sys.argv[2],dtype=['U25', '<i8'], usecols=(0, 1))

    w1 = []
    b1 = []
    w2 = []
    b2 = []

    #Reading model file
    with open(sys.argv[3], "r") as file:
        for i, line in enumerate(file):
            if i >= 1 and i <= 30:
                w1.append(map(float, line.split(' ')))
            if i >= 32 and i <= 61:
                b1.append(map(float, line.split(' ')))
            if i >= 63 and i <= 66:
                w2.append(map(float, line.split(' ')))
            if i >= 68 and i <= 71:
                b2.append(map(float, line.split(' ')))

    w1 = np.array(w1)
    b1 = np.reshape(np.array(b1), 30)
    w2 = np.array(w2)
    b2 = np.reshape(np.array(b2), 4)

    # Forward propogation
    z1 = np.dot(test, w1.T) + b1
    a1 = sigmoidvalue(z1)
    z2 = np.dot(a1, w2.T) + b2
    a2 = sigmoidvalue(z2)

    count = 0
    output = np.argmax(a2, axis=1) * 90

    file = open("output.txt", "w")
    for i in range(len(output)):
        if output[i] == test_label[i][1]:
            count = count + 1
        file.write(" ".join([str(test_label[i][0]), str(output[i]), "\n"]))

    print "Accuracy = ", float(count) * 100 / len(test_label)


if sys.argv[1] == "train":
    if sys.argv[4] == "adaboost":
        train_adaboost()
    elif sys.argv[4] == "nearest":
        train_nearest()
    elif sys.argv[4] == "nnet":
        train_nnet()
    elif sys.argv[4] == "best":
        train_nnet()

if sys.argv[1] == "test":
    if sys.argv[4] == "adaboost":
        test_adaboost()
    elif sys.argv[4] == "nearest":
        test_nearest()
    elif sys.argv[4] == "nnet":
        test_nnet()
    elif sys.argv[4] == "best":
        test_nnet()
