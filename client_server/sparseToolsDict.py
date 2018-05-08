########################################################################
#       Sparse version of the library tools with dictionnaries.        #
########################################################################

import math



########################################################################
# Different operations that are usefull to write the stochastic gradient
# descent algorithm and others.
########################################################################

# Compute the scalar product between to vectors spVec1 and spVec2.


def sparse_dot(spVec1, spVec2):
    return sum([val * spVec2.get(key, 0) for key, val in spVec1.items() if (key != -1)])


# Each component of vect is modified by func.
def sparse_map(func, spVec):
    return {key: (val if key == -1 else func(val)) for key, val in spVec.items()}


# Compute the sum, componentwise between u and v.
def sparse_vsum(spVec1, spVec2):
    similar_keys = set(spVec1.keys()) & set(spVec2.keys())
    summ = {k: spVec1[k] + spVec2[k] for k in similar_keys}

    sp1_only_keys = set(spVec1.keys()) - similar_keys
    sp1 = {k: spVec1[k] for k in sp1_only_keys}

    sp2_only_keys = set(spVec2.keys()) - similar_keys
    sp2 = {k: spVec2[k] for k in sp2_only_keys}

    summ.update(sp1)
    summ.update(sp2)

    return summ


# Compute the substraction spVec1-spVec2 element-wise as vsum(u,mutl(-1,v))
def sparse_vsous(spVec1, spVec2):
    def opp(x):
        return -x
    return sparse_vsum(spVec1, sparse_map(opp, spVec2))


# u is divided by v componentwise
def sparse_vdiv(spVec1, spVec2):
    if (len(spVec1) != len(spVec2)):
        print('You try to use sparse_vdiv with vectors of different sizes.')
    else:
        similar_keys = set(spVec1.keys()) & set(spVec2.keys())
        div = {k: (spVec1[k] / spVec2[k]) for k in similar_keys}
        return div


# each component of u is multiplied by a
def sparse_mult(a,spVec1):
    multVec = {}
    for key, value in spVec1.items():
        if (key != -1):
            multVec[key] = a*value
        else:
            multVec[key] = value
    return multVec


# Retrieve the label of a dictionary if it exists

def take_out_label(spVec):
    r = dict(spVec)
    try:
        del r[-1]
    except KeyError:
        pass
    return r

# Retrieve the data with key hypPlace if it exists

def take_out(spVec,hypPlace):
    r = dict(spVec)
    try:
        del r[hypPlace]
    except KeyError:
        pass
    return r


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

# Convert a dictionnary into a string.

def dict2str(dict):
    if (type(dict) != str):
        txt = ""
        for key, value in dict.items():
            txt += str(key)+":"+str(value)+"<->"
        return txt[0:-3]
    else:
        return dict

# Convert a string vector into a dictionnary

def str2dict(s):
    v = s.split("<->")
    dict = {}
    for e in v:
        kv = e.split(":")
        dict[float(kv[0])] = float(kv[1])
    return dict



# Convert a data set (list of dictionaries) to a


def datadict2Sstr(data):
    dataStr =  ""
    for d in data:
        label = d.get(-1,0)
        example = take_out_label(d)
        dstr = str(label) + "<|>" + dict2str(example)
        dataStr += dstr + "<<->>"
    return dataStr[0:-5]

# Convert a data string into a data set (lists)

def str2datadict(strData):
    frame = []
    datastr = strData.split("<<->>")
    for dstr in datastr:
        lab_ex = dstr.split("<|>")
        label = float(lab_ex[0])
        dict = str2dict(lab_ex[1])
        dict[-1] = label
        frame.append(dict)
    return frame












####################################################################
# Treat the data : normalise and center each example. This way,
# examples with huge norms don't have more importance than the
# others.
####################################################################



# Process the treatment on the set data.

def dataPreprocessing(data,hypPlace):

    n = len(data)

    # Computation of the average for each variable in the example
    moy = {}

    for k in range(n):
        example = take_out_label(data[k])
        moy = sparse_vsum(moy,example)

    def div(x):
        return float(x)/n

    moy = sparse_map(div,moy)


    #print('')
    #print("Le vecteur des moyennes est : " +str(moy))



    # Computation of the deviation
    sigma = {}

    for k in range(n):

        example = take_out_label(data[k])
        dictDiff = sparse_vsous(example,moy)

        def square(x):
            return x*x

        dictSquare = sparse_map(square,dictDiff)

        sigma = sparse_vsum(dictSquare,sigma)


    def sig(x):
        return math.sqrt((1./n)*x)

    sigma = sparse_map(sig,sigma)

    #print("Le vecteur des ecarts-types est : " + str(sigma))
    #print('')


    for k in range(n):
        label = data[k].get(-1,0)
        example = take_out_label(data[k])
        sous = sparse_vsous(example,moy)
        treatedEx = sparse_vdiv(sous,sigma)
        treatedEx[-1] = label
        treatedEx[hypPlace] = -1
        data[k] = treatedEx

    return data

