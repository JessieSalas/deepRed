import numpy as np
from PIL import Image
import os
import random

path = os.getcwd()+"/waldo"
nonpath = os.getcwd()+"/non_waldo"

XY = []

c=0
for f in range(0,6):
    im = Image.open(path+"/solo_waldo"+str(f) + ".png")
    im = im.convert(mode="L")
    iar = np.asarray(im)
    #dont care about transparency
    """
    iz = np.zeros((iar.shape[0],iar.shape[1],3))
    for i in range(iar.shape[0]):
        for j in range(iar.shape[1]):
            iz[i][j] = iar[i][j][:3]
    """
    final = []
    for row in iar:
        final.extend([ pix/255.0 for pix in row] )
    final = np.array(final)
    XY.append((final, 1))

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
    im = Image.open(nonpath+"/non_waldo"+str(i) + ".png")
    im = im.convert(mode="L")
    iar = np.asarray(im)
    """
    #dont care about transparency
    iz = np.zeros((iar.shape[0],iar.shape[1],3))
    for i in range(iar.shape[0]):
        for j in range(iar.shape[1]):
            iz[i][j] = iar[i][j][:3]
    """
    final = []
    for row in iar:
        final.extend([pix/255.0 for pix in row])
    final = np.array(final)
    XY.append((final,0))

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
y=[]

for tup in XY:
    #print tup
    print tup[0]
    X.append(tup[0])
    y.append(np.array([tup[1]]))


######NEWWORK

X = np.array(X)
y = np.array(y)


def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
        
    return 1/(1+np.exp(-x))
"""
X = np.array([[0,0,1,0],
              [0,1,1,1],
              [1,0,1,1],
              [1,1,1,0]])

y = np.array([[0],
              [1],
              [1],
              [0]])
"""
np.random.seed(1)
print X.shape
print y.shape
# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((X.shape[1],X.shape[0])) - 1
syn1 = 2*np.random.random((y.shape[0],y.shape[1])) - 1

for j in xrange(60000):
    
    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    
    # how much did we miss the target value?
    l2_error = y - l2
    
    if (j% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)
    
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
