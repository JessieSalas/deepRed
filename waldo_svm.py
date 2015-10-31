from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn import svm
from itertools import chain

path = os.getcwd()+"/waldo"
nonpath = os.getcwd()+"/non_waldo"
XY = []

c=0
for f in range(0,6):
    im = Image.open(path+"/solo_waldo"+str(f))
    iar = np.asarray(im)
    print iar
    exit()
    l = []
    for i in range(0, len(iar[0])):
        l.append(tuple(iar[0][i]))
    
    XY.append((l, 1))

    """
    if c==0:
        print iar[0][0]
        print iar[0][1]
        #print iar
        temp = np.asarray(XY[0][0][0])
        print temp.shape
        c+=1
    """

for i in range(0,6):
    im = Image.open(nonpath+"/non_waldo"+str(i))
    print im[0]
    exit()
    iar = np.asarray(im)
    l = []
    for i in range(0, len(iar[0])):
        l.append(sum(iar[0][i]))
    """
    if c==1:
        print iar
        c+=1
    """
    XY.append((l,0))
"""
for i in range(1, 89):
    try:
        im = Image.open(os.getcwd()+"/waldono/no"+str(i)+".png")
        iar = np.asarray(im)
        l = []
        for i in range(0, len(iar[0])):
            l.append(sum(iar[0][i]))
        #if c==2:
        #    print iar
        #    c+=1
        XY.append((l,0))
    except Exception as e:
        print(e)
"""
print(XY[0][0])
#random.shuffle(XY)

X=[]
Y=[]

for tup in XY:
    #print tup
    X.append(tup[0])
    Y.append(tup[1])
print X
print Y[0]
clf = svm.SVC(gamma = 0.001, C=100)

x = X[:176]
y = Y[:176]
exit()
clf.fit(x,y)

for i in range(1, 20):
    print(clf.predict(X[-i]))
    print(Y[-i])
