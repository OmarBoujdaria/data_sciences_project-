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
    if (len(v) > 1):
        for i in range(len(v)):
            v[i] = float(v[i])
    return v





def data2Sstr(data):
    n = len(data[0][1])
    dataStr =  ""
    for i in range(len(data)):
        dstr = str(data[i][0])+'<|>'
        for j in range(n):
            if (j == 0):
                dstr += str(data[i][1][j])
            else:
                dstr += '<->' + str(data[i][1][j])
        if (i == 0):
            dataStr += dstr
        else:
            dataStr += '<<->>' + dstr
    return dataStr



def str2data(strData):
    frame = strData.split("<<->>")
    print(frame[0])
    for i in range(len(frame)):
        labex = frame[i].split("<|>")
        label = float(labex[0])
        example = labex[1].split("<->")
        for k in range(len(example)):
            example[k] = float(example[k])
        frame[i] = [label,example]
    return frame
