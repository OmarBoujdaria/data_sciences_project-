import pickle
import math

# NECESSARY: run in python3 for good  performance on dict iterations
# cf : https://docs.quantifiedcode.com/python-anti-patterns/readability/not_using_items_to_iterate_over_a_dictionary.html


# ---------------------------------------------------------------------------------
#  this code goes to the main so we read the file only once
with open('Data12000', 'rb') as f:
    data = pickle.load(f)


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