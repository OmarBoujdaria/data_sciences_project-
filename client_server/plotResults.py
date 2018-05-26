####### Plot results of the synchronous/asynchronous SGD according to records in a text file

import matplotlib.pyplot as plt

############### OPENING OF THE FILES AND PREPARATION ##############

choice = 0

while ((choice != 1) & (choice != 2)):
    print("What would you plot : \n      1) Dense ? \n      2) Sparse ?")
    choice = input()

if (choice == 1):
    filePath = '/home/kiwi974/cours/epfl/system_for_data_science/project/client_server/synchronous.txt'
else:
    filePath = '/home/kiwi974/cours/epfl/system_for_data_science/project/client_server/asynchronous.txt'

file = open(filePath, 'r')

components = []
errorsTab = []

# Extract data of the file

for line in file:
    data = line.split("<nbCompo>")
    components.append(int(data[0]))
    err = data[1].split((", "))
    n = len(err)
    errors = []
    for k in range(n):
        if (k == 0):
            errors.append(float(err[k][1:]))
        elif (k == (n-1)):
            errors.append(float(err[k][:-2]))
        else:
            errors.append((float(err[k])))
    errorsTab.append(errors)
file.close()

print(errors)

# Plot data

colors = ['firebrick', 'darkorange', 'rebeccapurple', 'gold', 'darkgreen', 'dodgerblue', 'magenta','brown']

n = len(components)

###################################################################


figure = plt.figure(figsize=(10, 10))
splitComp = 0
plt.plot([k + splitComp for k in range(len(errorsTab[n - 1]) - splitComp)], errorsTab[n - 1][splitComp:],colors[n - 1], label="Error for classic SGD (all components, " + str(len(errorsTab[n-1])) + " iterations).")
for i in range(n-1):
    plt.plot([k+splitComp for k in range(len(errorsTab[i])-splitComp)], errorsTab[i][splitComp:], colors[i], label="Error for "+str(components[i])+" components chosen in topk (" + str(len(errorsTab[i])) + " iterations).")
    plt.xlabel("Server iterations.")
    plt.ylabel("Error")
    plt.title("Dense data : learning rate multiplied by 0.9 at each server iteration.")
    plt.legend()
plt.show()