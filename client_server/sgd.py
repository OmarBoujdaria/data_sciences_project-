import random
#import matplotlib.pyplot as plt

########################################################################
# Scalar product between two vectors.
########################################################################

def ps(u,v):
    res = 0
    for i in range(len(u)):
        res += u[i]*v[i]
    return res


def mult(a,vect):
    res = []
    for i in range(len(vect)):
        res.append(a*vect[i])
    return res


def vsum(u,v):
    res = []
    for i in range(len(u)):
        res.append(u[i]+v[i])
    return res


def sous(u,v):
    res = []
    for i in range(len(u)):
        res.append(u[i]-v[i])
    return res


########################################################################
# Generation of data to test stochastic gradient descent.
# Format : [label : Int, List of int]
# Input : -nbData : number of data generated.
#         -nbVar : number of variables per data. 
########################################################################

d = 1
u = [1,1]

def generateData(nbData):

    " Generation des donnees "
    
    A= []
    B = []
    
    nbExemples = 0
    nbRejetes = 0

    absA, absB = [], []
    ordA, ordB = [], []

    cardA = 0
    cardB = 0

    while (nbExemples < nbData):
        a = random.randint(0,100)/10
        b = random.randint(0,100)/10
        dist = abs((ps(u,[a,b]))/(ps(u,u))-5)
        valide = (d==0) or ((d!=0) & (dist >= d))
        if (a > 10-b) & valide:
            A.append([1,[a,b]])
            absA.append(a)
            ordA.append(b)
            nbExemples += 1
            cardA += 1
        elif (b < 10-b) & valide:
            B.append([-1,[a,b]])
            absB.append(a)
            ordB.append(b)
            nbExemples += 1
            cardB += 1
        else:
            nbRejetes += 1

    #plt.scatter(absA,ordA,s=10,c='r',marker='*')
    #plt.scatter(absB,ordB,s=10,c='b',marker='o')
    #plt.plot([0,10],[10,0],'orange')
    #plt.show()

    return(A+B)


########################################################################
# Stochastic gradient descent.
# Input : -data : the data set to sample and on which learn.
#         -w : the current vector of parameters.
#         -numSamples : the number of samples we want for the learning.
#         -step : departure step of the gradient descent.
#         -l : multiplier of parameter's vector's norm.
# Output : -w : the vector of parameters according to a random sample
#               of data.
########################################################################


#Remarks : data will be read in a textfile when we will have it


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




def descent(data,w,numSamples,step,l):

    dataSample,sampleSize = sample(data,numSamples)

    d = der_error(w,l,dataSample,sampleSize)
    w = sous(w,mult(step,d))

    return w




#coord1 = random.random()
#coord2 = random.random()

#w = sgd(data,[coord1,coord2],1000,0.01,0.5)
#print(w)


""" Computation of the eror given by the SVM on a sample. """

def error(w,l,sample,sampleSize):
    norm =  (l/2)*ps(w,w)
    sum = 0
    for i in range(sampleSize):
        label = sample[i][0]
        example = sample[i][1]
        sum += max(0,1-label*ps(w,example))
    res = norm + sum
    return res


""" Computation of the derivation of the error on a sample. """

def der_error(w,l,sample,sampleSize):
    d = mult(l, w)
    sum = [0 for i in range(len(w))]
    for i in range(sampleSize):
        label = sample[i][0]
        example = sample[i][1]
        if (label*ps(w,example) <= 1):
            sum = vsum(sum,mult(label,example))
    res = vsum(d,sum)
    return res

