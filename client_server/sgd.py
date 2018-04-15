import random
import tools
#import matplotlib.pyplot as plt








########################################################################
# Generation of data to test stochastic gradient descent. One can
# visualize the data : remove the comments on the import of matplotlib
# (line 2) and one the plot (lines 102 to 105). We generate points in
# the square [0,10]x[0,10], the hyperplan is y = 10 - x.

# Input :
#       -nbData : number of generated data.

# Output :
#       -trainingSet : a set of tuples (label,example) which is a
#        training. The format of this set is
#        List(label : Int, example : List(float)) -> label = +1 or -1.
########################################################################


# d is the double of the distance of each point in the square to the
# separator hyperplan.
d = 1

# u is a hyperplan's orthogonal vector.
u = [1,1]

def generateData(nbData):

    # A and B denote each a different class, respectively associated
    # to the labels 1 and -1.
    A= []
    B = []

    # Number of examples we kept for our training set.
    nbExamples = 0

    # Number of data we rejected because they are not at the good
    # distance of the hyperplan.
    nbRejeted = 0

    absA, absB = [], []
    ordA, ordB = [], []

    cardA = 0
    cardB = 0

    while (nbExamples < nbData):
        a = random.randint(0,100)/10
        b = random.randint(0,100)/10
        dist = abs((tools.ps(u,[a,b]))/(tools.ps(u,u))-5)
        valide = (d==0) or ((d!=0) & (dist >= d))
        if (a > 10-b) & valide:
            A.append([1,[a,b]])
            absA.append(a)
            ordA.append(b)
            nbExamples += 1
            cardA += 1
        elif (b < 10-b) & valide:
            B.append([-1,[a,b]])
            absB.append(a)
            ordB.append(b)
            nbExamples += 1
            cardB += 1
        else:
            nbRejeted += 1

    #plt.scatter(absA,ordA,s=10,c='r',marker='*')
    #plt.scatter(absB,ordB,s=10,c='b',marker='o')
    #plt.plot([0,10],[10,0],'orange')
    #plt.show()

    trainingSet = A+B

    return(trainingSet)







########################################################################
# Sample a set in a subset of numSamples elements.
# Input :
#       -set : the set to sample.
#       -numSamples : the number of elements of set that we want in
#        the subset.
# Output :
#       -dataSample : the subset (sample of set).
#       -sampleSize : the size of the sample set (which is equal to
#        numSamples.
########################################################################


def sample(set,numSamples):
    n = len(set)
    dataSample = []
    cpyData = list(set)

    # Sample the data set on which complete the learning phase
    sampleSize = 0
    while (sampleSize < numSamples):
        s = random.randint(0, n - sampleSize - 1)
        dataSample.append(cpyData[s])
        del cpyData[s]
        sampleSize += 1

    return dataSample, sampleSize




########################################################################
# Computation of the error given by the SVM on a sample.
# Input :
#       -w : the vector of parameters.
#       -l : the depreciation factor of the SVM.
#       -sample : the training set.
#       -sampleSize : number of examples in sample, i.e. the size of
#        this set.
# Output :
#       -cost : the cost computed on sample.
########################################################################

def error(w,l,sample,sampleSize):
    norm =  (l/2)*tools.ps(w,w)
    sum = 0
    for i in range(sampleSize):
        label = sample[i][0]
        example = sample[i][1]
        sum += max(0,1-label*tools.ps(w,example))
    cost = norm + sum
    return cost



########################################################################
# Computation of the derivation of the error on a sample.
# Input :
#       -w : the vector of parameters.
#       -l : the depreciation factor of the SVM.
#       -sample : the training set.
#       -sampleSize : number of examples in sample, i.e. the size of
#        this set.
# Output :
#       -dcost : the value of the derivative of the cost computed on
#        sample.
########################################################################

def der_error(w,l,sample,sampleSize):
    d = tools.smult(l, w)
    sum = [0 for i in range(len(w))]
    for i in range(sampleSize):
        label = sample[i][0]
        example = sample[i][1]
        if (label*tools.ps(w,example) <= 1):
            sum = tools.vsum(sum,tools.smult(label,example))
    dcost = tools.vsum(d,sum)
    return dcost





########################################################################
# Stochastic gradient descent.
# Input : -data : the data set to sample and on which learn.
#         -w : the current vector of parameters.
#         -numSamples : the number of samples we want for the learning.
#         -step : departure step of the gradient descent.
#         -l : depreciation factor of the SVM.
# Output : -w : the vector of parameters according to a random sample
#               of data.
########################################################################



def descent(data,w,numSamples,step,l):

    # Sample of the data set.
    dataSample,sampleSize = sample(data,numSamples)

    # Derivative of the cost evaluated on dataSample.
    d = der_error(w,l,dataSample,sampleSize)

    # Modification of the parameter vector w.
    w = tools.vsous(w,tools.smult(step,d))

    return w





