# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the gRPC route guide client."""

from __future__ import print_function

import random

import grpc
import time
from random import randint

import route_guide_pb2
import route_guide_pb2_grpc

import sgd
import convertion


def vect2str(v):
    txt = ""
    n = len(v)
    for i in range(n):
        txt += str(v[i])
        if (i != (n-1)):
            txt += "<->"
    return txt



def str2vect(s):
	v = s.split("<->")
	return v



numSamples = 5


def guide_get_feature(stub):

    it = 1

    vect = stub.GetFeature(route_guide_pb2.Vector(poids='pret'))

    #vect is the data set
    dataSampleSet = convertion.str2data(vect.poids)

    #Get the departure vector
    vect = stub.GetFeature(route_guide_pb2.Vector(poids='getw0'))
    l = 0.5
    step = 0.05
    while (vect.poids != 'stop'):
        print("iteration : " + str(it))
        nw = sgd.descent(dataSampleSet,convertion.str2vect(vect.poids),numSamples,step,l)
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
