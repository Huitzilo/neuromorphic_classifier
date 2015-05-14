#global variables
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-s", "--station", dest="workstation",
                  help="spikey workstation to use", default="station666")
parser.add_option("-n", "--num_data_samples", dest="num_data_samples", type="int",
                  help="number of data samples from each class to use", default=200)
parser.add_option("-d", "--digits", dest="digits_txt", help="digits to be used",
                  default="5,7")



options, args = parser.parse_args()

workstation = options.workstation #default: 'station603'
num_data_samples = options.num_data_samples # default: 200
digits = [int(x) for x in options.digits_txt.split(',')]

# imports
import numpy
import sys
try:
	import pyNN.hardware.stage1 as p
except ImportError, e:
	print('ImportError: {}.\n'.format(e.message))
	print("Failed to import the hardware PyNN module. \n")
	print("The current version of this script needs the Spikey hardware.")
	print("I'll be glad to help you getting it to run a simulator (i.e. NEST).\n" +\
		  "If you're interested, open an issue at \n" +\
		  "https://github.com/Huitzilo/neuromorphic_classifier .")
	sys.exit(1)
import neuclar.network_controller as netcontrol
from  neuclar.network_config import hardware_sparse_config as config
config['network']['decision_pops'] = '2'
import neuclar.data.mnist as mnist
import neuclar.vrconvert as vrconvert

# doing the stuff
p.setup(workStationName=workstation)

# load data
training_data, training_labels = mnist.get_training_data(digits, num_data_samples)
testing_data, testing_labels = mnist.get_training_data(digits, num_data_samples)

# convert data with VRs
training_data_vr = vrconvert.vrconvert(training_data, vrconvert.mnist_samplepoints)
testing_data_vr = vrconvert.vrconvert(testing_data, vrconvert.mnist_samplepoints)

# make data right format
training_patterns = [("mnist_%d"%i, training_data_vr[i], training_labels[i])
                                             for i in range(len(training_labels))]
testing_patterns = [("mnist_%d"%i, testing_data_vr[i], testing_labels[i])
                                             for i in range(len(testing_labels))]
class_ids = list(set(training_labels))

# set up classifier network
bc = netcontrol.BrainController(p, config)
bc.brain.wire_AL_to_MBext()
bc.brain.reinit_random_weights()

# train the network
for tp in training_patterns:
    bc.learn_pattern(tp, class_ids)
test_results = [bc.test_pattern(tp) for tp in testing_patterns]

# assess percent correct
ind_pred = [numpy.argmax(tr) for tr in test_results]
ind_target = [class_ids.index(l) for l in testing_labels]
correct = [i==il for i,il in zip(ind_pred,ind_target)]

num_correct = sum(correct)
num_total = len(correct)
percent_correct = float(num_correct)/float(num_total)

print "Correctly classified %d out of %d (%.2f correct)"%(num_correct, num_total, percent_correct)
