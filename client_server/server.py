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
"""The Python implementation of the gRPC route guide server."""

from concurrent import futures

import time
import waiting

import grpc
import random
import route_guide_pb2
import route_guide_pb2_grpc
import math
import threading

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

import sgd
from convertion import *

nbClients = 2
paramSize = 5





def merge(vectors):
    n = len(vectors[0]) #all the vectors have the same size
    vmoy = [0 for i in range(n)]
    for i in range(n):
        summ = 0 
        for k in range(nbClients):
            summ += float(vectors[k][i])
        vmoy[i] = summ/nbClients
    return vmoy


data = sgd.generateData(30)

w0 = [random.random() for k in range(len(data[0][1]))]

nbMaxCall = 20

class RouteGuideServicer(route_guide_pb2_grpc.RouteGuideServicer):

    def __init__(self):
        self.iterator = 0
        self.enter_condition = (self.iterator == nbClients)
        self.exit_condition = (self.iterator == 0)
        self.vectors = []
        self.epoch = 0
        self.oldParam = w0
        self.nbCall = 0
        self.printerThreadName = ''

    def GetFeature(self, request, context):

        self.iterator += 1
        self.vectors.append(str2vect(request.poids))
    
        self.enter_condition = (self.iterator == nbClients)
        waiting.wait(lambda : self.enter_condition)

        self.printerThreadName = threading.current_thread().name
            
        if (request.poids == 'pret'):
            vector = data2Sstr(data)
        elif (request.poids == 'getw0'):
            vector = vect2str(w0)
        else :
            vector = merge(self.vectors)
            diff = sgd.sous(self.oldParam,vector)
            normDiff = math.sqrt(sgd.ps(diff,diff))
            if ((normDiff <= 10**(-3)) or (self.nbCall > nbMaxCall)):
                vector = 'stop'
            else:
                vector = vect2str(vector)
                 
        self.iterator -= 1

        self.exit_condition = (self.iterator == 0)
        waiting.wait(lambda : self.exit_condition)

        realComputation = (request.poids != 'pret') and (request.poids != 'getw0') and (vector != 'stop')

        if (realComputation):
            self.oldParam = str2vect(vector)

        ###################### PRINT OF THE CURRENT STATE ######################
        if (threading.current_thread().name == self.printerThreadName):
            print('')
            print('############################################################')
            if (self.epoch == 0):
                print('# We went the data to the clients.')
            else:
                print('# We performed the epoch : ' + str(self.epoch) + '.')
            if (realComputation or (self.epoch == 1)):
                print('# The merged vector is : ' + vector + '.')
            print('############################################################')
            print('')
            self.epoch += 1
        ############################### END OF PRINT ###########################
        
        self.vectors = []
        waiting.wait(lambda : (self.vectors == []))

        self.nbCall += 1
        
        time.sleep(1)
        return route_guide_pb2.Vector(poids=vector)



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    route_guide_pb2_grpc.add_RouteGuideServicer_to_server(
        RouteGuideServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
