####### Plot results of the synchronous/asynchronous SGD according to records in a text file

import matplotlib.pyplot as plt

############### OPENING OF THE FILES AND PREPARATION ##############

choice = 0

while ((choice != 1) & (choice != 2)):
    print("What would you plot : \n      1) Synchronous ? \n      2) Asynchronous ?")
    choice = input()

if (choice == 1):
    filePathTraining = '/home/kiwi974/cours/epfl/system_for_data_science/project/client_server/synchronousT.txt'
    filePathValidation = '/home/kiwi974/cours/epfl/system_for_data_science/project/client_server/synchronousV.txt'
else:
    filePathTraining = '/home/kiwi974/cours/epfl/system_for_data_science/project/client_server/asynchronousT.txt'
    filePathValidation = '/home/kiwi974/cours/epfl/system_for_data_science/project/client_server/asynchronousV.txt'

file = open(filePathTraining, 'r')

times = 0
errorsTraining = []
errorsValidation = []

# Extract training errors of the file

data = file.read().split("<nbCompo>")
durationT = float(data[0])
err = data[1].split((", "))
n = len(err)
for k in range(n):
    if (k == 0):
        errorsTraining.append(float(err[k][1:]))
    elif (k == (n-1)):
        errorsTraining.append(float(err[k][:-2]))
    else:
        errorsTraining.append((float(err[k])))
file.close()

# Extract validation errors of the file

file = open(filePathValidation, 'r')

data = file.read().split("<nbCompo>")
durationV = float(data[0])
err = data[1].split((", "))
n = len(err)
for k in range(n):
    if (k == 0):
        errorsValidation.append(float(err[k][1:]))
    elif (k == (n-1)):
        errorsValidation.append(float(err[k][:-2]))
    else:
        errorsValidation.append((float(err[k])))
file.close()


# Plot data

colors = ['firebrick', 'darkorange', 'rebeccapurple', 'gold', 'darkgreen', 'dodgerblue', 'magenta','brown']


###################################################################


figure = plt.figure(figsize=(10, 10))
plt.plot([k for k in range(len(errorsTraining))], errorsTraining, colors[0], label="Learning curve. Computation time = " + str(durationT))
plt.plot([k for k in range(len(errorsValidation))], errorsValidation, colors[5], label="Validation curve. Computation time = " + str(durationT))
plt.xlabel("Server iterations.")
plt.ylabel("Error")
plt.title("Dense data : learning rate multiplied by 0.9 at each server iteration.")
plt.legend()
plt.show()