import sys
import random
import math

'''Get feature data from file as a matrix with a row per data instance'''
def getFeatureData(featureFile):
    x=[]
    dFile = open(featureFile, 'r')
    for line in dFile:
        row = line.split()
        rVec = [float(item) for item in row]
        x.append(rVec)
    dFile.close()
    return x

'''Get label data from file as a dictionary with key as data instance index and value as the class index'''
def getLabelData(labelFile):
    lFile = open(labelFile, 'r')
    lDict = {}
    for line in lFile:
        row = line.split()
        if int(row[0]) == 0:
            lDict[int(row[1])] = -1
        else:
            lDict[int(row[1])] = int(row[0])
    lFile.close()
    return lDict

def dotProduct(u,v):
    sum = 0
    for i in range(len(u)):
        sum += u[i] * v[i]
    return sum

def norm(v):
    return math.sqrt(dotProduct(v,v)) 

def program():
    data = getFeatureData("data1.data")
    trainlabels = getLabelData("data1.labels")
    rows = len(data)
    cols = len(data[0])

    ##initialize the value of w
    w = []
    for j in range(cols):
        w.append(0)
        w[j] = (0.02 * random.uniform(0,1)) - 0.01

    ##gradient_descent
    eta = 0.0001
    ##calculate error outside the loop
    error=0.0
    for i in range (rows):
        if(trainlabels[i] != None):
            error += (-trainlabels[i]+ dotProduct(w,data[i]) )**2

    #initialize flag and iteration parameters
    flag = 0
    k=0

    while(flag != 1):
        k+=1
        df = []
        for i in range(cols):
            df.append(0)

        for i in range(rows):
            if(trainlabels.get(i) != None):
                r = dotProduct(w, data[i])
                for j in range (cols):
                    #calculate gradient
                    df[j] += (-trainlabels[i] + r) * data[i][j]

        for j in range(cols):
            w[j] = w[j] - eta*df[j]
            
            ##computing error
            curr_error = 0
            for i in range (rows):
                if(trainlabels.get(i) != None):
                    curr_error += ( -trainlabels[i] + dotProduct(w,data[i]) )**2
            print(error,k)
            if error - curr_error < 0.001:
                flag = 1
            error = curr_error

        ### print error
    ## calculate error difference:

    #print("count",k)
    #print("w =",w)

    normw = 0
    for j in range((cols-1)):
        normw += w[j]**2

    wt=[]
    for i in range(len(w)-1):
        wt.append(w[i])
    
    normv = norm(wt)
    print("||w||=", normw, normv)

    d_origin = w[(len(w)-1)] / normw

    misses = 0
    for i in range(rows):
        if(trainlabels[i] != None):
            d_p = dotProduct(w, data[i])
            if(d_p > 0):
                print("1",i)
                if(trainlabels[i]!=1):
                    misses+=1
            else:
                print("0",i)
                if(trainlabels[i]!=0):
                    misses+=1
    print("Accuracy: ", (rows-misses)*100/rows)
program()
