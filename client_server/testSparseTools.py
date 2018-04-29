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

st_map_v = [[2,4],[3,7],[8,3]]   # = list(v) --> doesn't work, modify v too ???!!

print(id(st_map_v) == id(v))

def opp(x):
    return (-x)

print("v = " + str(v))

st.sparse_map(opp,st_map_v)

print("st_map_v = " + str(st_map_v))






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
st.sparse_vdiv(list(u),[1.0,3.5,0.5,1.5],4,4,s_div)

print("s_div = " + str(s_div))