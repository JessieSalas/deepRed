# Deep Red
Solving "Where's Waldo" with deep learning in the form of feedforward convolutional neural networks + clever preprocessing to increase our data set

Previous attempts at automatically solving Where's Waldo puzzles have failed to adhere to the standard rules of the puzzle and have not yet solved anything.
The previously best known solution uses a naive pattern-matching procedure to find waldo in an image given a cropped picture of his exact pose in that image. Implementations like this are trivial and equate to a simple linear search. 

The way a human is expected to solve a Where's Waldo puzzle is different; humans must use their memory of what waldo looks like, do mental transformations to account for differences in pose, and recognize him freely by scanning the image.


We implement a feedforward convolutional neural network to solve this. 
In optimization we use:
