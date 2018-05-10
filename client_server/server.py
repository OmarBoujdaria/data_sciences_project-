"""The Python implementation of the gRPC stochastic gradient descent server.
Note : it's necessary to download and install the package waiting of Python to execute that code. it permits to create
barriers for the threads at different point of the algorithm. while loops was not enough : we can explain the different
problems we had with that synchronization part during the meeting, and justify the use of the librairy. """

from concurrent import futures

import math
import time
import waiting

import grpc
import route_guide_pb2
import route_guide_pb2_grpc

import threading

import sgd
import sparseToolsDict as std

import pickle

_ONE_DAY_IN_SECONDS = 24 * 60 * 60

""" Define the number of clients you want to use."""
nbClients = 2



# Place of the constante 1 in each example : it
# permits to include the hyperplan constant to the
# vector of parameters.

hypPlace = 10**6


# Importation of the data

def treatData(data):
    for i in range(len(data)):
        if (data[i].get(-1,0) == [[1]]):
            data[i][-1] = 1
    return data

print("Starting of the server...")

with open('/home/kiwi974/cours/epfl/system_for_data_science/project/data/data6000new', 'rb') as f:
    data = treatData(pickle.load(f))

# Number of examples we want in our training set.
nbExamples = 200

# Number of examples we want in our testing set.
nbTestingData = 200

print("Building of the training set...")

# Define the training set.
trainingSet = data[:nbExamples]

print("Training data pre-processing...")

trainingSet = std.dataPreprocessing(trainingSet,hypPlace)

print("Building of the testing set...")

# Define the testing set.
testingSet = data[nbExamples:nbExamples+nbTestingData]

print("Testing data pre-processing")

testingSet = std.dataPreprocessing(testingSet, hypPlace)

# Pre-processing of the data (normalisation and centration).
# data = tools.dataPreprocessing(data)

# Initial vector to process the stochastic gradient descent :
# random generated.
w0 = {1: 0.21, 2: 0.75, hypPlace: 0.011}  # one element, to start the computation
normw0 = math.sqrt(std.sparse_dot(w0, w0))
nbParameters = len(trainingSet[0]) - 1  # -1 because we don't count the label

# Maximum number of epochs we allow.
nbMaxCall = 50

# The constant step to perform the gradient descent on the learning training.
step = 0.05

print("Server is ready !")

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
        # The final vector of parameters we find
        self.paramVector = {}
        # Error on the training set, computed at each cycle of the server
        self.trainingErrors = []
        # Error on the testing set, computed at each cycle of the server
        self.testingErrors = []

    def GetFeature(self, request, context):

        ######################################################################
        # Section 1 : wait for all the clients -> get their vectors and
        # appoint one of them as the printer.

        self.iterator += 1
        if (request.poids == "pret" or request.poids == "getw0"):
            self.vectors.append(request.poids)
        else:
            self.vectors.append(std.str2dict(request.poids))
        self.enter_condition = (self.iterator == nbClients)
        waiting.wait(lambda: self.enter_condition)

        self.printerThreadName = threading.current_thread().name

        ######################################################################

        ######################################################################
        # Section 2 : compute the new vector -> send the data, a merge of
        # all the vectors we got from the clients or the message 'stop' the
        # signal to the client that we converged.

        if (request.poids == 'pret'):
            vector = std.datadict2Sstr(trainingSet)
        elif (request.poids == 'getw0'):
            vector = std.dict2str(w0)
        else:
            grad_vector = std.merge(self.vectors,nbClients)
            vector = std.sparse_vsous(self.oldParam,grad_vector)
            diff = std.sparse_vsous(self.oldParam, vector)
            normDiff = math.sqrt(std.sparse_dot(diff, diff))
            normGradW = math.sqrt(std.sparse_dot(vector, vector))
            if ((normDiff <= 10 ** (-3)) or (self.epoch > nbMaxCall) or (normGradW <= 10 ** (-3) * normw0)):
                self.paramVector = vector
                vector = 'stop'
            else:
                vector = std.dict2str(vector)

        ######################################################################

        ######################################################################
        # Section 3 : wait that all the threads pass the computation area, and
        # store the new computed vector.

        self.iterator -= 1

        self.exit_condition = (self.iterator == 0)
        waiting.wait(lambda: self.exit_condition)

        realComputation = (request.poids != 'pret') and (request.poids != 'getw0') and (vector != 'stop')

        if (realComputation):
            self.oldParam = std.str2dict(vector)

        ######################################################################

        ###################### PRINT OF THE CURRENT STATE ######################
        if (threading.current_thread().name == self.printerThreadName):
            print('')
            print('############################################################')
            if (self.epoch == 0):
                print('# We sent the data to the clients.')
            else:
                print('# We performed the epoch : ' + str(self.epoch) + '.')
                if (vector == "stop"):
                    print("# The vector that achieve the convergence is : " + str(self.paramVector))
            if (realComputation or (self.epoch == 1)):
                # Compute the error made with that vector of parameters on the testing set
                self.testingErrors.append(sgd.error(self.oldParam, 0.1, testingSet, nbTestingData, hypPlace))
                self.trainingErrors.append(sgd.error(self.oldParam, 0.1, trainingSet, nbExamples, hypPlace))
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
        waiting.wait(lambda: (self.vectors == []))

        ######################################################################

        # time.sleep(1)
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
