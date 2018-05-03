########################################################################
#           Test the sparse tools library with dictionaries.           #
########################################################################

import sparseToolsDict as std



# Initialisation of the data

spV1 = {1:4, 3:5, 7:1, 9:10}
spV2 = {1:4, 2:3, 7:6, 9:2}
spV3 = {1:2, 3:5, 7:5, 9:2}

data = [spV1, spV2, spV3]
data[0][-1] = 1
data[1][-1] = -1
data[2][-1] = 1



print('')
print("############## Test of the sparse scalar product. ##############")
print('')


spdsp = std.sparse_dot(spV1, spV2)
print("spdsp = " + str(spdsp))


print('')
print("################ Test of the sparse sum. #################")
print('')


spdsu = std.sparse_vsum(spV1, spV2)
print("spdsum = " + str(spdsu))

empty = std.sparse_vsum({},spV1)
print("empty = " + str(empty))


print('')
print("############### Test of the sparse map. ###############")
print('')


def opp(x):
    return (-x)


spdmap = std.sparse_map(opp,spV1)
print("spdmap = " + str(spdmap))


print('')
print("############ Test of the sparse soustraction. #########")
print('')


spdsous = std.sparse_vsous(spV1,spV2)
print("spdsous = " + str(spdsous))


print('')
print("############ Test of the sparse division. ############")
print('')


spddiv = std.sparse_vdiv(spV1,spV2)
print("spddiv = " + str(spddiv))


print('')
print("############ Test of the elementwise multiplication. ############")
print('')


spdmult = std.sparse_mult(2,spV1)
print("spdmult = " + str(spdmult))

print('')
print("######### Test of the conversion dict -> str. ##########")
print('')


spddict2str = std.dict2str(spV1)
print("spV1 under string is : " + spddict2str)


print('')
print("######### Test of the conversion str -> dict. ##########")
print('')


spdstr2dict = std.str2dict(spddict2str)
print("spV1 under dict is : " + str(spdstr2dict))



print('')
print("######### Test of the conversion data -> str. ##########")
print('')


print("label of data1 = " + str(data[0].get(-1,0)))
print("label of data2 = " + str(data[1].get(-1,0)))
print("label of data3 = " + str(data[2].get(-1,0)))


print("data0 = " + str(data[0]))

spddatadict2str = std.datadict2Sstr(data)
print("data under string is : " + spddatadict2str)


print('')
print("######### Test of the conversion str -> data. ##########")
print('')

spdstr2datadict = std.str2datadict(spddatadict2str)
print("data under list of dictionaries is : " + str(spdstr2datadict))








print('')
print("########### Test of the data preprocessing ############")
print('')




e1 = [[1,2],[3,7],[7,1],[10,3]]
e2 = [[2,4],[3,7],[8,3]]
e3 = [[2,1],[3,7],[9,7]]
e4 = [[7,8],[10,4]]
e5 = [[1,2],[2,3],[3,4],[4,5],[5,6],[9,1]]

data = [[1,e1],[-1,e2],[1,e3],[1,e4],[-1,e5]]










