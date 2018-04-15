####################################################################
# A set of functions that are used among the different functions in
# sgd.py, client.py and server.py .
####################################################################


import math
import convertion




####################################################################
# Each element of the training set is a list of the form :
# List(label : int, example : List(float). In order to send and
# receive this data with the simplest service of gRPC (and this way
# avoid serialization), we need to convert this data in text and
# vice-versa. Each element of vectors are separated by '<->', a label
# and its example are separated bu '<|>' and at least two elements
# of the training set are separated by '<<->>'.
# Thus, we have to different string format :
#       -for a vector : [1,3,2,9] <=> '1<->3<->2<->9'.
#       -for a data set : [[1,[4,3,5]],[-1,[8,9,4]]] <=>
#                         '1<|>4<->3<->5<<->>-1<|>8<->9<->4'.
####################################################################

# Convert a vector (list) into a string.

def vect2str(v):
    txt = ""
    n = len(v)
    for i in range(n):
        txt += str(v[i])
        if (i != (n-1)):
            txt += "<->"
    return txt

# Convert a string vector into a vector (list)

def str2vect(s):
    v = s.split("<->")
    if (len(v) > 1):
        for i in range(len(v)):
            v[i] = float(v[i])
    return v



# Convert a data set (lists) into a string

def data2Sstr(data):
    n = len(data[0][1])
    dataStr =  ""
    for i in range(len(data)):
        dstr = str(data[i][0])+'<|>'
        for j in range(n):
            if (j == 0):
                dstr += str(data[i][1][j])
            else:
                dstr += '<->' + str(data[i][1][j])
        if (i == 0):
            dataStr += dstr
        else:
            dataStr += '<<->>' + dstr
    return dataStr

# Convert a data string into a data set (lists)

def str2data(strData):
    frame = strData.split("<<->>")
    print(frame[0])
    for i in range(len(frame)):
        labex = frame[i].split("<|>")
        label = float(labex[0])
        example = labex[1].split("<->")
        for k in range(len(example)):
            example[k] = float(example[k])
        frame[i] = [label,example]
    return frame








####################################################################
# Treat the data : normalise and center each example. It permits to
# seek for an hyperplan (SVM) which passes by 0.
####################################################################

# Auxiliary function that process the treatment on an example

def vectPreprocessing(ex):
    n = len(ex)
    moy = 0
    for i in range(len(ex)):
        moy = moy + ex[i]
    moy = moy/n
    sigma = 0
    for i in range(len(ex)):
        sigma = sigma + (ex[i] - moy)**2
    sigma = math.sqrt(sigma/(n-1))
    res = convertion.sous(ex,[moy for i in range(n)])
    res = convertion.div(res,[sigma for i in range(n)])
    #On ne conserve que 5 chiffres après la virgule
    for i in range(len(res)):
        res[i] = ((res[i]*100000)//1)/100000
    return res
