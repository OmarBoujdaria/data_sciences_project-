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
        print("res = " + str(res))
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

def sparse_map(func,u):
    for i in range(len(u)):
        u[i][1] = func(u[i][1])



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

    minus_cv = list(v)

    def opp(x):
        return (-x)

    sparse_map(opp,minus_cv)
    print("minus_cv = " + str(minus_cv))
    sparse_vsum(u,minus_cv,lenu,lenv,res)



# u is divided by v componentwise

def sparse_vdiv(cu,cv,lenu,lenv,res):

    if (lenu != lenv):
        print('You try to use sparse_vdiv with vectors of different sizes.')
    else :
        for i in range(lenu):
            res.append([cu[i][0],cu[i][1]/cv[i]])

















####################################################################
# Treat the data : normalise and center each example. It permits to
# seek for an hyperplan (SVM) which passes by 0.
####################################################################

# Auxiliary function that processes the treatment on an example.

def vectPreprocessing(ex,moy,sigma,lenmoy,lensigma):

    cex = list(ex)
    cmoy = list(moy)
    csigma = list(sigma)

    ex = []

    sparse_vsous(cex,cmoy,lenmoy,lensigma,ex)
    sparse_vdiv(cex,csigma,lenmoy,lensigma,ex)

    return ex



# Process the treatment on the hole data set.

def dataPreprocessing(data):

    n = len(data)

    # Computation of the average for each variable in the example
    moy = []

    for j in range(n):

        cmoy = list(moy)
        cex = data[j][1]
        moy = []
        sparse_vsum(cmoy,cex,len(moy),len(cex),moy)

        def div(x):
            return x/n

        sparse_map(div,moy)

    lenmoy = len(moy)


    # Computation of the deviation
    sigma = []

    for j in range(n):

        cdata = list(data[j][1])
        cmoy = list(moy)
        data_minus_moy = []
        sparse_vsous(cdata,cmoy,len(cdata),lenmoy,data_minus_moy)

        def square(x):
            return x*x

        sparse_map(square,data_minus_moy)

        csigma = list(sigma)
        sigma = []
        sparse_vsum(csigma,data_minus_moy,len(csigma),len(data_minus_moy),sigma)


    for k in range(n):
        data[k][1] = vectPreprocessing(data[k][1],moy,sigma)

    return data
