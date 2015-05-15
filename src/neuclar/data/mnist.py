# took file format from from http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
#to whom credit is due.
import os
import struct
import gzip
import numpy
import operator

def read_mnist(digits, dataset="training"):
    """
    Read the mnist data set. 
    Data set obtained from http://yann.lecun.com/exdb/mnist/ .
    
    Parameters: 
    digits - list of digits to read
    dataset - either "training" or "testing"
    """
    assert dataset in ['testing','training']
    filename_prefix = {'testing':'t10k', 'training':'train'}
    path = os.path.dirname(__file__)
    filename_images = os.path.join(path, '%s-images-idx3-ubyte.gz'%filename_prefix[dataset])
    filename_labels = os.path.join(path, '%s-labels-idx1-ubyte.gz'%filename_prefix[dataset])

    labelfile = gzip.open(filename_labels, 'rb')
    magic_nr, size = struct.unpack(">II", labelfile.read(8))
    labelstring = labelfile.read()
    labelfile.close()
    labels = numpy.fromstring(labelstring, dtype=numpy.int8)

    imagefile = gzip.open(filename_images, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", imagefile.read(16))
    imagestring = imagefile.read()
    imagefile.close()
    images = numpy.fromstring(imagestring, dtype=numpy.uint8)
    images = images.reshape((size, len(images)/size))

    index_list = [labels == digit for digit in digits ]
    indices = reduce(operator.or_, index_list)
    retimg = images[indices,:]
    retlab = labels[indices]

    return retimg, retlab

def get_training_data(digits, num_samples):
    """
    Read num_samples training instances of the digits from the MNIST database.

    Returns 2-tuple (vec, lab) with vec a num_samples X 784 array of pixel grey values,
    and lab an num_samples length array of labels.
    
    Parameters:
    digits - list of digits to read (list of int)
    num_samples - number of samples (int)
    """
    img,lab = read_mnist(digits, dataset='training')
    img = img[:num_samples,:]
    lab = lab[:num_samples]
    return img, lab
    
def get_testing_data(digits, num_samples):
    """
    Read num_samples testing instances of the digits from the MNIST database.

    Returns 2-tuple (vec, lab) with vec a num_samples X 784 array of pixel grey values,
    and lab an num_samples length array of labels.
    
    Parameters:
    digits - list of digits to read (list of int)
    num_samples - number of samples (int)
    """
    img,lab = read_mnist(digits, dataset='testing')
    img = img[:num_samples,:]
    lab = lab[:num_samples]
    return img, lab    


# unit test and example
if __name__ == '__main__':
    import pylab
    digits = [1,2,3,4]
    img,lab = read_mnist(digits)
    img = img[:200,:]
    lab = lab[:200]
    f = pylab.figure()
    for i in range(4):
	pylab.subplot(2,2,i)
	pylab.imshow(numpy.reshape(img[i,:], (28,28)), cmap='gray')
    import mdp
    wn = mdp.nodes.WhiteningNode(output_dim=2)
    wn.train(numpy.array(img,dtype=float))
    filt = wn.execute(numpy.array(img,dtype=float))
    f = pylab.figure()
    for d in digits:
      pylab.plot(filt[lab==d,0], filt[lab==d,1], '.')
      pylab.hold(True)
    pylab.show()
