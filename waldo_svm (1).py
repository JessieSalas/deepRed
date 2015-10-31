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
for f in range(0,49):
    im0 = Image.open(path+"/solo_waldo"+str(f)+".png")
    im0 = im0.convert('L')
    y = np.asarray(im0.getdata(),dtype=np.float64).reshape((im0.size[1],im0.size[0]))
    XY.append((y,1))

for i in range(0,62):
    im0 = Image.open(nonpath+"/non_waldo"+str(i)+".png")
    im0 = im0.convert('L')
    y = np.asarray(im0.getdata(),dtype=np.float64).reshape((im0.size[1],im0.size[0]))
    XY.append((y,0))

for i in range(1, 89):
    try:
        im0 = Image.open(os.getcwd()+"/waldono/no"+str(i)+".png")
        im0 = im0.convert('L')
        y = np.asarray(im0.getdata(),dtype=np.float64).reshape((im0.size[1],im0.size[0]))
        XY.append((y,0))
    except Exception as e:
        print(e)

random.shuffle(XY)

X=[]
Y=[]

for tup in XY:
    X.append(tup[0])
    Y.append(tup[1])

print len([item for sublist in X[0] for item in sublist])

clf = svm.SVC(gamma = 0.001, C=100)

x = X[:176]
y = Y[:176]

x = np.asarray(x)
y = np.asarray(y)

clf.fit(x,y)

for i in range(1, 20):
    print(clf.predict(X[-i]))
    print(Y[-i])
