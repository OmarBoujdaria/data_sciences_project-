########################################################################
#                Test the sparse tools library.                        #
########################################################################
import copy

import sparseTools as st



# Initialisation of the data

u = [[1,2],[3,7],[7,1],[10,3]]
v = [[2,4],[3,7],[8,3]]

lenu = 4
lenv = 3



print('')
print("############## Test of the sparse scalar product. ##############")
print('')

st_ps = 0
st.sparse_ps(list(u),list(v),lenu,lenv,st_ps)

print("st_ps = " + str(st_ps))







print('')
print("################ Test of the sparse sum. #################")
print('')

st_sum = []
st.sparse_vsum(list(u),list(v),lenu,lenv,st_sum)

print("st_sum = " + str(st_sum))





print('')
print("############### Test of the sparse map. ###############")
print('')


def opp(x):
    return (-x)

st_map = []

st.sparse_map(opp,v,st_map)

print("st_map_v = " + str(st_map))






print('')
print("############ Test of the sparse soustraction. #########")
print('')

st_sous = []
st.sparse_vsous(list(u),list(v),lenu,lenv,st_sous)

print("st_sous = " + str(st_sous))




print('')
print("############ Test of the sparse division. ############")
print('')

s_div = []
st.sparse_vdiv(list(u),[[1,1.0],[3,3.5],[7,0.5],[10,1.5]],4,4,s_div)

print("s_div = " + str(s_div))


print('')
print("#####################################")
print("u = " + str(u))
print("v = " + str(v))
print("#####################################")
print('')
print('')
print('')
print('')
print("########### Test of the data porcessing ############")


e1 = [[1,2],[3,7],[7,1],[10,3]]
e2 = [[2,4],[3,7],[8,3]]
e3 = [[2,1],[3,7],[9,7]]
e4 = [[7,8],[10,4]]
e5 = [[1,2],[2,3],[3,4],[4,5],[5,6],[9,1]]

data = [[1,e1],[-1,e2],[1,e3],[1,e4],[-1,e5]]

print("Processed data = " + str(st.dataPreprocessing(data)))