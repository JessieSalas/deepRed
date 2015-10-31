import numpy as np

from PIL import Image
import os
import random
import glob
import pickle

def loadImages(image_path,num_images,label):
    XY = []
    for filename in os.listdir(image_path):
        if filename.split(".")[-1] == "png":
            #for f in range(0,num_images):
            im = Image.open(image_path + "/" + filename)
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
            XY.append((final, label))
            
            """
                if c==0:
                print iar[0][0]
                print iar[0][1]
                #print iar
                temp = np.asarray(XY[0][0][0])
                print temp.shape
                c+=1
            """
    return XY

"""
    for i in range(0,6):
    im = Image.open(nonpath+"/non_waldo"+str(i))
    im = im.convert(mode="L")
    iar = np.asarray(im)
    #dont care about transparency
    iz = np.zeros((iar.shape[0],iar.shape[1],3))
    for i in range(iar.shape[0]):
    for j in range(iar.shape[1]):
    iz[i][j] = iar[i][j][:3]
    final = []
    for row in iar:
    final.extend([pix/256.0 for pix in row])
    final = np.array(final)
    XY.append((final,0))
    
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
XY = []
c=0
nonpath = os.getcwd()+"/non_waldo"
path = os.getcwd()+"/waldo"

images = loadImages(path,True,1)
images.extend(loadImages(nonpath,True,0))

XY = images
#print(XY[0][0])
#random.shuffle(XY)

X=[]
y=[]
print XY[0]
for tup in XY:
    #print tup
    #print tup[0]
    X.append(np.array(tup[0]))
    y.append(np.array([tup[1]]))


######NEWWORK
random.shuffle(X)
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




alphas = [0.1]
hiddenSize = 32

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
"""
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])
"""
for alpha in alphas:
    print "\nTraining With Alpha:" + str(alpha)
    np.random.seed(1)
    
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((X.shape[1],hiddenSize)) - 1
    synapse_1 = 2*np.random.random((hiddenSize,1)) - 1
    
    for j in xrange(60000):
        
        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0,synapse_0))
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        layer_2_error = layer_2 - y
        

        if (j% 10000) == 0:
            print "Error after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error)))

            print(layer_2)
    
        layer_2_delta = layer_2_error*sigmoid_output_to_derivative(layer_2)
        
        layer_1_error = layer_2_delta.dot(synapse_1.T)
        
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
        synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))
    #outputting the data
    with open("layer0.pickle","wb") as out1:
        pickle.dump(layer_0,out1, protocol=pickle.HIGHEST_PROTOCOL)

    with open("layer1.pickle","wb") as out2:
        pickle.dump(layer_1,out2, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open("layer2.pickle","wb") as out3:
        pickle.dump(layer_2,out3, protocol=pickle.HIGHEST_PROTOCOL)
