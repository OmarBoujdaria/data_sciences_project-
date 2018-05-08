import pickle
import math

# NECESSARY: run in python3 for good  performance on dict iterations
# cf : https://docs.quantifiedcode.com/python-anti-patterns/readability/not_using_items_to_iterate_over_a_dictionary.html


def treatData(data):
    for i in range(len(data)):
        if (data[i].get(-1,0) == [[1]]):
            data[i][-1] = 1
    return data

# ---------------------------------------------------------------------------------
#  this code goes to the main so we read the file only once
with open('/home/kiwi974/cours/epfl/system_for_data_science/project/data/data12000', 'rb') as f:
    data = treatData(pickle.load(f))



def see_label(data):
    dict = {}
    for i in range(len(data)):
        val = data[i].get(-1,0)
        if (val == [[1]]):
            val = 1
        if (val in dict):
            dict[val] += 1
        else:
            dict[val] = 1
    return dict



count = see_label(data)
print(count)
print("nbExamples = " + str(count.get(-1,0) + count.get(1,0)))


# i is the row, j is the collumn
#
# The Data variables is a list of dictionaries
# data[i],
#		 is a dictinary containgi
# new function:ng the sparse representation of the ith row
#
# data[i].get(j,0).
#		 to get the elem at jth column
#		 to get the label of the row, set j=-1
#		 comment :  the zero parameter is what is return if key not found in dict
#                   this is consistent with sparse representation
# ---------------------------------------------------------------------------------

def take_out_label(spVec):
    r = dict(spVec)
    try:
        del r[-1]
    except KeyError:
        pass

    return r