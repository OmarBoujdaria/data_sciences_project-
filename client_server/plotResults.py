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


times = 0
absSet = []
errorsTraining = []
errorsValidation = []

# Extract training errors of the file

file = open(filePathTraining, 'r')
data = file.read().split("<nbCompo>")
duration = float(data[0])
data[1] = data[1][2:]
data[1] = data[1][:-3]
err = data[1].split(("), ("))

n = len(err)
for k in range(n):
    f = err[k].split(", ")
    errorsTraining.append(float(f[0]))
    absSet.append(int(f[1]))
file.close()

# Extract validation errors of the file

file = open(filePathValidation, 'r')
data = file.read().split("<nbCompo>")
durationT = float(data[0])
data[1] = data[1][2:]
data[1] = data[1][:-3]
err = data[1].split(("), ("))

n = len(err)

for k in range(n):
    f = err[k].split(", ")
    errorsValidation.append(float(f[0]))
file.close()


# Plot data

colors = ['firebrick', 'darkorange', 'rebeccapurple', 'gold', 'darkgreen', 'dodgerblue', 'magenta','brown']


###################################################################


figure = plt.figure(figsize=(10, 10))
plt.plot(absSet, errorsTraining, colors[0], label="Learning curve. Computation time = " + str(duration))
plt.plot(absSet, errorsValidation, colors[5], label="Validation curve. Computation time = " + str(duration))
plt.xlabel("Server iterations.")
plt.ylabel("Error")
plt.title("Dense data : learning rate multiplied by 0.9 at each server iteration.")
plt.legend()
plt.show()