#global variables
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-s", "--station", dest="workstation",
                  help="spikey workstation to use", default="station666")
parser.add_option("-n", "--num_data_samples", dest="num_data_samples", type="int",
                  help="total number of data samples to use", default=200)
parser.add_option("-d", "--digits", dest="digits_txt", help="digits to be used",
                  default="5,7", type="string")
parser.add_option("-o", "--output_file", dest='output_file', type="string",
		  help="put detailed results in this file", default=None)
parser.add_option("-r", "--retrain_VRs", dest='retrain_VRs', action='store_true',
		  help='train a new Neural Gas instead of reusing default VRs',
		  default=False)

options, args = parser.parse_args()

workstation = options.workstation 
num_data_samples = options.num_data_samples
digits = [int(x) for x in options.digits_txt.split(',')]
output_file_name = options.output_file
retrain = options.retrain_VRs

# imports
import numpy
import sys
import time
import logging
logging.basicConfig()
lg = logging.getLogger('mnist_classifier_on_spikey')
lg.setLevel(logging.INFO)
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
if retrain:
	posfilename = "vrpos-{}-{}.npy".format(['{}'.format(d) for d in digits],
					       time.strftime("%Y-%m-%d-%H-%M-%S"))
	lg.info("computing new VR positions, storing them to {}".format(posfilename))
	vrs = vrconvert.NeuralGasSampler()
	vrs.train_sampler(numpy.array(training_data, dtype=float), vrconvert.neural_gas_parameters)
	numpy.save(posfilename, vrs.samplepoints)
	training_data_vr = vrs.sample_data(training_data)
	testing_data_vr = vrs.sample_data(testing_data)
else:
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
percent_correct = 100.*float(num_correct)/float(num_total)

print "Correctly classified {} out of {} ({:.2f} % correct)".format(num_correct,
								    num_total,
								    percent_correct)

if not(output_file_name is None):
	import os
	if output_file_name in os.listdir('.'):
		print('Output file {} exists.'.format(output_file_name))
		import tempfile
		fdll, output_file_name = tempfile.mkstemp(suffix=output_file_name, prefix="", dir='.')
		print("using {} instead.".format(output_file_name))
	fd = open(output_file_name, 'w')
	resultlines = []
	resultlines.append("{}\n\n".format(time.asctime()))
	resultlines.append("Digits: {}\n".format(digits))
	resultlines.append("max. num. samples: {}\n\n".format(num_data_samples))
	resultlines.append("{:.2f} % correct ({} out of {})\n".format(percent_correct,
								      num_correct,
								      num_total))
	resultlines.append("\n\n")
	resultlines.append("target\tpredicted\n")
	resultlines.append("------\t---------\n")
	for t,p in zip(testing_labels, test_results):
		resultlines.append("{}\t{}\n".format(t, class_ids[numpy.argmax(p)]))
	fd.writelines(resultlines)
	fd.close()
				   
