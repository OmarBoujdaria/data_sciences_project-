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

times = 0
errors = []

# Extract data of the file

data = file.read().split("<nbCompo>")
duration = float(data[0])
err = data[1].split((", "))
n = len(err)
for k in range(n):
    if (k == 0):
        errors.append(float(err[k][1:]))
    elif (k == (n-1)):
        errors.append(float(err[k][:-2]))
    else:
        errors.append((float(err[k])))
file.close()

print(errors)

# Plot data

colors = ['firebrick', 'darkorange', 'rebeccapurple', 'gold', 'darkgreen', 'dodgerblue', 'magenta','brown']


###################################################################


figure = plt.figure(figsize=(10, 10))
plt.plot([k for k in range(len(errors))], errors, colors[0], label="Learning curve. Computation time = " + str(duration))
plt.xlabel("Server iterations.")
plt.ylabel("Error")
plt.title("Dense data : learning rate multiplied by 0.9 at each server iteration.")
plt.legend()
plt.show()