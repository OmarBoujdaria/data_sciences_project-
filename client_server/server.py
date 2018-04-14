"""The Python implementation of the gRPC stochastic gradient descent server.
Note : it's necessary to download and install the package waiting of Python to execute that code. it permits to create
barriers for the threads at different point of the algorithm. while loops was not enough : we can explain the different
problems we had with that synchronization part during the meeting, and justify the use of the librairy. """

from concurrent import futures

import time
import waiting

import grpc
import random
import route_guide_pb2
import route_guide_pb2_grpc
import math
import threading


import sgd
from convertion import *


_ONE_DAY_IN_SECONDS = 24*60*60

""" Define the number of clients you want to use."""
nbClients = 2



def merge(vectors):
    n = len(vectors[0]) #all the vectors have the same size
    vmoy = [0 for i in range(n)]
    for i in range(n):
        summ = 0 
        for k in range(nbClients):
            summ += float(vectors[k][i])
        vmoy[i] = summ/nbClients
    return vmoy


# Number of examples we want in our training.
nbExamples = 30

# Set of generated data.
data = sgd.generateData(nbExamples)

# Initial vector to process the stochastic gradient descent :
# random generated.
w0 = [random.random() for k in range(len(data[0][1]))]

# Maximum number of epochs we allow.
nbMaxCall = 20





class RouteGuideServicer(route_guide_pb2_grpc.RouteGuideServicer):



    """ We define attributes of the class to perform the computations."""
    def __init__(self):
        # An iterator that will count the number of clients that contact the
        # server at each epoch.
        self.iterator = 0
        # A barrier condition to be sure that every waited client contacted
        # the server before to start the GetFeature method (kind of join).
        self.enter_condition = (self.iterator == nbClients)
        # An other barrier condition, that acts like a join on the threads to.
        self.exit_condition = (self.iterator == 0)
        # A list to store all the vectors sent by each client at each epoch.
        self.vectors = []
        # The current epoch (0 -> send the data to the clients).
        self.epoch = 0
        # The previous vecor of parameters : the last that had been sent.
        self.oldParam = w0
        # The name of one of the thread executing GetFeature : this one, and
        # only this one will something about the state of the computation in
        # the server.
        self.printerThreadName = ''



    def GetFeature(self, request, context):

        ######################################################################
        # Section 1 : wait for all the clients -> get their vectors and
        # appoint one of them as the printer.

        self.iterator += 1
        self.vectors.append(str2vect(request.poids))
    
        self.enter_condition = (self.iterator == nbClients)
        waiting.wait(lambda : self.enter_condition)

        self.printerThreadName = threading.current_thread().name

        ######################################################################

        ######################################################################
        # Section 2 : compute the new vector -> send the data, a merge of
        # all the vectors we got from the clients or the message 'stop' the
        # signal to the client that we converged.

        if (request.poids == 'pret'):
            vector = data2Sstr(data)
        elif (request.poids == 'getw0'):
            vector = vect2str(w0)
        else :
            vector = merge(self.vectors)
            diff = sgd.sous(self.oldParam,vector)
            normDiff = math.sqrt(sgd.ps(diff,diff))
            if ((normDiff <= 10**(-3)) or (self.epoch > nbMaxCall)):
                vector = 'stop'
            else:
                vector = vect2str(vector)

        ######################################################################

        ######################################################################
        # Section 3 : wait that all the threads pass the computation area, and
        # store the new computed vector.
                 
        self.iterator -= 1

        self.exit_condition = (self.iterator == 0)
        waiting.wait(lambda : self.exit_condition)

        realComputation = (request.poids != 'pret') and (request.poids != 'getw0') and (vector != 'stop')

        if (realComputation):
            self.oldParam = str2vect(vector)

        ######################################################################

        ###################### PRINT OF THE CURRENT STATE ######################
        if (threading.current_thread().name == self.printerThreadName):
            print('')
            print('############################################################')
            if (self.epoch == 0):
                print('# We sent the data to the clients.')
            else:
                print('# We performed the epoch : ' + str(self.epoch) + '.')
            if (realComputation or (self.epoch == 1)):
                print('# The merged vector is : ' + vector + '.')
            if (self.epoch == nbMaxCall):
                print('We performed the maximum number of iterations.')
                print('The descent has been stopped.')
            print('############################################################')
            print('')
            self.epoch += 1
        ############################### END OF PRINT ###########################

        ######################################################################

        ######################################################################
        # Section 4 : empty the storage list of the vectors, and wait for all
        # the threads.

        self.vectors = []
        waiting.wait(lambda : (self.vectors == []))

        ######################################################################

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
