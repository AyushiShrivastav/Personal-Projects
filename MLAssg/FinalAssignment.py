import sys

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

'''Get label data from file as a dictionary with key as data instance index
and value as the class index
'''
def getLabelData(labelFile):
    lFile = open(labelFile, 'r')
    lDict = {}
    for line in lFile:
        row = line.split()
        lDict[int(row[1])] = int(row[0])
    lFile.close()
    return lDict

#Calculating mean of the dataset passed as an arguement

def mean(dataset):
    sum=0.000
    n=0
    for mini in dataset:
        for item in mini:
            sum+=item
            n+=1
    return(sum/n)

#Calculating variance of the dataset passed as an arguement

def variance(dataset, mean):
    n=0
    var=0.000
    for mini in dataset:
        for item in mini:
            var+=(item-mean)**2
            n+=1
    return(var/n)

def classification(test, mean, variance):
    cl=0.0000
    for item in test:
        cl+=((item-mean)**2/variance)
    return cl

def separateClass():
    cp0=0.0000
    cp1=0.0000
    dataset = []
    classset = {}
    class0=[]
    class1=[]
    m0=0.000
    m1=0.000
    file1 = sys.argv[1]    #getFeatureData("C:\\Users\\CZL\\Desktop\\breast_cancer.data")
    file2 = sys.argv[2]    #getLabelData("C:\\Users\\CZL\\Desktop\\breast_cancer.labels")
    dataset = getFeatureData(file1) 
    classset = getFeatureData(file2) 
    i = 0
    n0=0
    n1=0
    while i < len(classset):
      if classset[i] == 0:
          class0.append(dataset[i])
      else:
          class1.append(dataset[i])
      i += 1
    i=0
    
    m0=mean(class0)
    m1=mean(class1)
    #print "Means", m0, m1

    var0=variance(dataset, m0)
    var1=variance(dataset, m1)
    #print "Variance", var0, var1

    #Testing the training dataset and predicting labels. 

    for mini in dataset:
        cp0=classification(mini, m0, var0)
        cp1=classification(mini, m1, var1)
        print cp0, cp1
        if(cp0>cp1):
            print "1", i
        else:
            print "0", i
        i+=1
        
def main():
    separateClass()

main()
