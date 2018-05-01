########################################################################
#             Sparse version of the library tools.                     #
########################################################################

import math




########################################################################
# Different operations that are usefull to write the stochastic gradient
# descent algorithm and others.
########################################################################


# Compute the scalar product between to vectors u and v.

def sparse_ps(cu,cv,lenu,lenv,res):

    if  ((lenu == 0) or (lenv == 0)):
        return res
    else:
        #print("####")
        if (cu[0][0] < cv[0][0]):
            del cu[0]
            lenu -= 1
        elif (cu[0][0] > cv[0][0]):
            del cv[0]
            lenv -= 1
        else :
            res += cu[0][1]*cv[0][1]
            del cu[0]
            del cv[0]
            lenu -= 1
            lenv -= 1

        #print("cu = " + str(cu) + " of length " + str(lenu))
        #print("cv = " + str(cv) + " of length " + str(lenv))
        #print("ps = " + str(res))

        return sparse_ps(cu,cv,lenu,lenv,res)



# Each component of vect is modified by func.

def sparse_map(func,u,res):
    for i in range(len(u)):
        res.append([u[i][0],func(u[i][1])])



# Compute the sum, componentwise between u and v.

def sparse_vsum(cu,cv,lenu,lenv,res):

    def add_list(u,v):
        for i in range(len(v)):
            u.append(v[i])

    if (lenu == 0):
        add_list(res,cv)
    elif (lenv == 0):
        add_list(res,cu)
    else:
        if (cu[0][0] == cv[0][0]):
            res.append([cu[0][0],cu[0][1] + cv[0][1]])
            del cu[0]
            del cv[0]
            lenu -= 1
            lenv -= 1
        elif (cu[0][0] < cv[0][0]):
            res.append(cu[0])
            del cu[0]
            lenu -= 1
        else:
            res.append(cv[0])
            del cv[0]
            lenv -= 1
        sparse_vsum(cu,cv,lenu,lenv,res)




# sous(u,v) = vsum(u,mutl(-1,v))

def sparse_vsous(u,v,lenu,lenv,res):

    minus_cv = []

    def opp(x):
        return (-x)

    sparse_map(opp,v,minus_cv)
    sparse_vsum(u,minus_cv,lenu,lenv,res)



# u is divided by v componentwise

def sparse_vdiv(cu,cv,lenu,lenv,res):

    def add_list(u,v):
        for i in range(len(v)):
            u.append(v[i])

    if (lenu == 0):
        add_list(res,[])
    elif (lenv == 0):
        add_list(res,[])
        print("v is empty before u : STRANGE. In sparse_div.")
    else:
        if (cu[0][0] == cv[0][0]):
            res.append([cu[0][0],cu[0][1]/cv[0][1]])
            del cu[0]
            del cv[0]
            lenu -= 1
            lenv -= 1
        elif (cu[0][0] < cv[0][0]):
            # In theory, in our application case (process data), this
            # case should never happen
            print("Somethong strange happenned in sparse_div, probably during dataProcessing.")
            res.append(cu[0])
            del cu[0]
            lenu -= 1
        else:
            del cv[0]
            lenv -= 1
        sparse_vdiv(cu,cv,lenu,lenv,res)









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

def sparse_vect2str(v):
    txt = ""
    n = len(v)
    for i in range(n):
        txt += str(v[i])
        if (i != (n-1)):
            txt += "<->"
    return txt

# Convert a string vector into a vector (list)

def sparse_str2vect(s):
    v = s.split("<->")
    if (len(v) > 1):
        for i in range(len(v)):
            v[i] = float(v[i])
    return v



# Convert a data set (lists) into a string

def sparse_data2Sstr(data):
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

def sparse_str2data(strData):
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

# Auxiliary function that processes the treatment on an example.

def vectPreprocessing(ex,moy,sigma,lenmoy,lensigma):

    cex = list(ex)
    cmoy = list(moy)
    csigma = list(sigma)

    ex_minus_moy = []
    ex_div_sigma = []

    sparse_vsous(cex,cmoy,len(cex),lenmoy,ex_minus_moy)
    sparse_vdiv(ex_minus_moy,csigma,len(ex_minus_moy),lensigma,ex_div_sigma)

    return ex_div_sigma



# Process the treatment on the hole data set.

def dataPreprocessing(data):

    n = len(data)

    # Computation of the average for each variable in the example
    moy = []

    for j in range(n):

        cmoy = list(moy)
        cex = list(data[j][1])
        temp_moy = []

        sparse_vsum(cmoy,cex,len(moy),len(cex),temp_moy)

        moy = list(temp_moy)

    def div(x):
        return x/n

    moy_pond = list(moy)
    moy = []
    sparse_map(div,moy_pond,moy)

    lenmoy = len(moy)


    print('')
    print("Le vecteur des moyennes est : " +str(moy))

    # Computation of the deviation
    sigma = []

    for j in range(n):

        cdata = list(data[j][1])
        cmoy = list(moy)
        data_minus_moy = []
        sparse_vsous(cdata,cmoy,len(cdata),lenmoy,data_minus_moy)

        def square(x):
            return x*x

        square_minus = []

        sparse_map(square,data_minus_moy,square_minus)

        csigma = list(sigma)
        temp_sigma = []
        sparse_vsum(csigma,square_minus,len(csigma),len(square_minus),temp_sigma)
        sigma = temp_sigma

    print("Le vecteur des écarts-types est : " + str(sigma))
    print('')


    for k in range(n):
        data[k][1] = vectPreprocessing(data[k][1],moy,sigma,lenmoy,len(sigma))

    return data
