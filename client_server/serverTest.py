"""The Python implementation of the gRPC stochastic gradient descent server.
Note : it's necessary to download and install the package waiting of Python to execute that code. It permits to create
barriers for the threads at different point of the algorithm. While loops was not enough : we can explain the different
problems we had with that synchronization part during the meeting, and justify the use of the librairy. """

from concurrent import futures

import math
import time
import waiting

import grpc
import route_guide_pb2
import route_guide_pb2_grpc

import threading

import sparseToolsDict as std

import time
import pickle
import sgd

_ONE_DAY_IN_SECONDS = 24 * 60 * 60






############ Define the number of clients you want to use. ############
nbClients = 2




############ Place of the constante 1 in each example : it permits to   ############
############ include the hyperplan constant to the vector of parameters ############

hypPlace = 10**6




############ Loading of the data ############

def treatData(data):
    for i in range(len(data)):
        if (data[i].get(-1,0) == [[1]]):
            data[i][-1] = 1
    return data

print("Starting of the server...")

with open('/home/kiwi974/cours/epfl/system_for_data_science/project/data/data6000new', 'rb') as f:
    data = treatData(pickle.load(f))





############ Definition of the parameters of the algorithm ############

# Number of examples we want in our training set.
nbExamples = 2000

# Number of chunks created to send data and size of each one of them
chunkSize = 1000
nbChunks = nbExamples//chunkSize + 1

# Number of samples we want for each training subset client
numSamples = 100

# The depreciation of the SVM norm cost
l = 0.01

# Maximum number of server iterations we allow.
nbMaxCall = 20

# The step of the descent
step = 1

# Number of examples we want in our testing set.
nbTestingData = 30

# Constants to test the convergence
c1 = 10**(-8)
c2 = 10**(-8)



############ Preprocessing of the data ############

print("Building of the training set...")

# Define the training set.
trainingSet = data[:nbExamples]

print("Training data pre-processing...")

#trainingSet = std.dataPreprocessing(trainingSet,hypPlace)

print("Building of the testing set...")

# Define the testing set.
testingSet = data[nbExamples:nbExamples+nbTestingData]

print("Testing data pre-processing")

#testingSet = std.dataPreprocessing(testingSet, hypPlace)




############ Computation of the initial departure vector and its gradient ############

# Initial vector to process the stochastic gradient descent :
# random generated.
w0 = {1: 0.21, 2: 0.75, hypPlace: 0.011}  # one element, to start the computation

print("Computation of the initial gradient")
gW0 = sgd.der_error(w0,l,trainingSet,nbExamples)
normGW0 = math.sqrt(std.sparse_dot(gW0,gW0))
nbParameters = len(trainingSet[0]) - 1  # -1 because we don't count the label





############ Way to work ############
way2work = "async"

# File path where record training erros
if (way2work == "sync"):
    filePath = '/home/kiwi974/cours/epfl/system_for_data_science/project/client_server/synchronous.txt'
else:
    filePath = '/home/kiwi974/cours/epfl/system_for_data_science/project/client_server/asynchronous.txt'



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
        # Step of the descent
        self.step = step
        # Keep all the merged vectors
        self.merged = [w0]
        # Keep the starting time
        self.startTime= 0



    def GetFeature(self, request, context):

        ######################################################################
        # Section 1 : wait for all the clients -> get their vectors and
        # appoint one of them as the printer.

        if (way2work == "sync" or ((way2work=="async") and (self.epoch == 0)) ):
            self.iterator += 1
            if (request.poids == "pret" or request.poids == "getw0" or request.poids[:5] == "chunk"):
                self.vectors.append(request.poids)
            else:
                self.vectors.append(std.str2dict(request.poids.split("<delay>")[0]))
            self.enter_condition = (self.iterator == nbClients)
            waiting.wait(lambda : self.enter_condition)

            self.printerThreadName = threading.current_thread().name

            if ((threading.current_thread().name == self.printerThreadName) & (self.epoch == 2)):
                ############ Starting of the timer to time the run ############
                self.startTime = time.time()

        ######################################################################

        ######################################################################
        # Section 2 : compute the new vector -> send the data, a merge of
        # all the vectors we got from the clients or the message 'stop' the
        # signal to the client that we converged.

        normDiff = 0
        normGradW = 0
        normPrecW = 0

        if (request.poids == 'pret'):
            vector = str(nbChunks) + "<depre>" + str(l) + "<samples>" + str(numSamples)
        elif (request.poids[:5] == 'chunk'):
            chunk = request.poids.split("<nb>")
            chunk = int(chunk[1])
            vector = std.datadict2Sstr(trainingSet[(chunk-1)*chunkSize:chunk*chunkSize])
        elif (request.poids == 'getw0'):
            vector = std.dict2str(w0)
        else :
            if (way2work == "sync"):
                gradParam = std.mergeSGD(self.vectors)
                if (self.epoch == 2):
                    self.normGradW0 = math.sqrt(std.sparse_dot(gradParam, gradParam))
                normGradW = math.sqrt(std.sparse_dot(gradParam, gradParam))
                gradParam = std.sparse_mult(self.step, gradParam)
                vector = std.sparse_vsous(self.oldParam, gradParam)
            else:
                info = request.poids.split("<delay>")
                grad_vector = std.str2dict(info[0])
                if (self.epoch == 2):
                    self.normGradW0 = math.sqrt(std.sparse_dot(grad_vector, grad_vector))
                normGradW = math.sqrt(std.sparse_dot(grad_vector, grad_vector))
                wt = std.str2dict(info[1])
                vector = std.asynchronousUpdate(self.oldParam,grad_vector,wt,l,self.step)

            ######## NORMALIZATION OF THE VECTOR OF PARAMETERS #########
            normW = math.sqrt(std.sparse_dot(vector,vector))
            vector = std.sparse_mult(1./normW,vector)

            ############################################################
            diff = std.sparse_vsous(self.oldParam,vector)
            normDiff = math.sqrt(std.sparse_dot(diff,diff))
            normPrecW = math.sqrt(std.sparse_dot(self.oldParam, self.oldParam))
            if ((normDiff <= c1*normPrecW) or (self.epoch > nbMaxCall) or (normGradW <= c2*normGW0)):
                self.paramVector = vector
                vector = 'stop'
            else:
                vector = std.dict2str(vector)

        ######################################################################

        ######################################################################
        # Section 3 : wait that all the threads pass the computation area, and
        # store the new computed vector.

        realComputation = (request.poids != 'pret') and (request.poids != 'getw0') and (vector != 'stop') and (request.poids[:5] != 'chunk')

        if (way2work == "sync" or ((way2work=="async") and (self.epoch == 0))):
            self.iterator -= 1

            self.exit_condition = (self.iterator == 0)
            waiting.wait(lambda : self.exit_condition)

        if (realComputation):
            self.oldParam = std.str2dict(vector)

        ######################################################################

        ###################### PRINT OF THE CURRENT STATE ######################
        ##################### AND DO CRITICAL MODIFICATIONS ####################
        if ((threading.current_thread().name == self.printerThreadName) & (way2work=="sync") or (way2work=="async")):

            endTime = time.time()
            duration = endTime - self.startTime

            if (vector == 'stop'):
                print("The server ran during : " + str(duration))

            std.printTraceRecData(self.epoch, vector, self.testingErrors, self.trainingErrors, normDiff, normGradW, normPrecW, normGW0,realComputation, self.oldParam,trainingSet, testingSet, nbTestingData, nbExamples,c1,c2,l, duration, filePath)

            self.merged.append(self.oldParam)
            if (realComputation):
                self.epoch += 1
                self.step *= 0.9
            ############################### END OF PRINT ###########################



            ######################################################################
            # Section 4 : empty the storage list of the vectors, and wait for all
            # the threads.

            self.vectors = []
            waiting.wait(lambda : (self.vectors == []))

            ######################################################################

        #time.sleep(1)
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

