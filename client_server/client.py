"""The Python implementation of the gRPC stochastic gradient descent server client."""

from __future__ import print_function

import random

import grpc
import time
from random import randint

import route_guide_pb2
import route_guide_pb2_grpc

import sgd
import convertion


# We define here the number of samples we want for each training subset.
numSamples = 5


def guide_get_feature(stub):

    # A variable to count the number of iteration of the client, which must coincide with the epoch in the server.
    it = 1

    # We make a first call to the server to get the data : after that call, vect is the data set. Then we store it.
    vect = stub.GetFeature(route_guide_pb2.Vector(poids='pret'))
    dataSet = vect.poids

    # This second call serves to get the departure vector. We store it, to eventually reuse it.
    vect = stub.GetFeature(route_guide_pb2.Vector(poids='getw0'))
    departureVector = vect

    # The depreciation of the SVM norm cost
    l = 0.5

    # The constant step to perform the gradient descent on the learning training.
    step = 0.05

    while (vect.poids != 'stop'):

        print("iteration : " + str(it))

        # We sample the data set to get a training subset.
        dataSampleSet = convertion.str2data(dataSet)

        # Gradient descent on the sample.
        nw = sgd.descent(dataSampleSet,convertion.str2vect(vect.poids),numSamples,step,l)

        # The result is sent to the server.
        vect.poids = convertion.vect2str(nw)
        vect = stub.GetFeature(route_guide_pb2.Vector(poids=vect.poids))

        it += 1
        time.sleep(1)
    print(vect)





def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = route_guide_pb2_grpc.RouteGuideStub(channel)
    print("-------------- GetFeature --------------")
    guide_get_feature(stub)


if __name__ == '__main__':
    run()
