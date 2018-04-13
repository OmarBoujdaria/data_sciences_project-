import pickle


with open ('Data12000', 'rb') as f:
    Data = pickle.load(f)


#le zero  en deuxieme parametre est ce qui doit etre retourné si la key n'est pas dans le Dict Data. 
#le label se trouve sous la key -1, pour le reste on ne reprensente que les index d'elements non nuls
def DataAccessor(i,key) :
    return Data[i].get(key, 0)