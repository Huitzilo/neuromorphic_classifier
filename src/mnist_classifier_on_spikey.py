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
parser.add_option("--save_spiketrains", dest="save_spiketrains", 
                  type='string', default='no_store',
                  help="specify a file in which to store the spike trains.\n" + 
                       "The spiketrains will be stored in a file named \n" + 
                       "'spiketrains.pkl' as a pickled dictionary,\n"+
                       "with keys 'train' and 'test', each containing a list of\n" +
                       "dictionaries that containing the actual spike trains.\n")

options, args = parser.parse_args()

workstation = options.workstation 
num_data_samples = options.num_data_samples
digits = [int(x) for x in options.digits_txt.split(',')]
output_file_name = options.output_file
retrain = options.retrain_VRs
save_spiketrains = options.save_spiketrains

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
config['network']['decision_pops'] = '{}'.format(len(digits))
# have to reduce the neuron count in the decision pop to fit in five digits, 
#config['network']['num_dec_neurons'] = '6'
#config['network']['num_inh_dec_neurons'] = '6'
import neuclar.data.mnist as mnist
import neuclar.vrconvert as vrconvert

start_time = time.time()

# doing the stuff
p.setup(workStationName=workstation)
setup_time = time.time()

# load data
training_data, training_labels = mnist.get_training_data(digits, num_data_samples)
testing_data, testing_labels = mnist.get_training_data(digits, num_data_samples)
load_data_time = time.time()

# convert data with VRs
if retrain:
	posfilename = "vrpos-{}-{}.npy".format("".join(['{}'.format(d) for d in digits]),
					       time.strftime("%Y-%m-%d-%H-%M-%S"))
	lg.info("computing new VR positions, storing them to {}".format(posfilename))
	vrs = vrconvert.NeuralGasSampler()
	vrs.train_sampler(numpy.array(training_data, dtype=float), vrconvert.neural_gas_parameters)
	numpy.save(posfilename, vrs.samplepoints)
	training_data_vr = vrs.sample_data(training_data)
	testing_data_vr = vrs.sample_data(testing_data)
else:
	vrposfilename = "vrpos-{}_{}.npy".format("".join(["{}".format(d) for d in digits]), num_data_samples)
	samplepoints = numpy.load(vrposfilename)
	training_data_vr = vrconvert.vrconvert(training_data, samplepoints)
	testing_data_vr = vrconvert.vrconvert(testing_data, samplepoints)

# make data right format
training_patterns = [("mnist_%d"%i, training_data_vr[i], training_labels[i])
                                             for i in range(len(training_labels))]
testing_patterns = [("mnist_%d"%i, testing_data_vr[i], testing_labels[i])
                                             for i in range(len(testing_labels))]
class_ids = list(set(training_labels))

convert_data_time = time.time()

if save_spiketrains:
    import cPickle
    #initialise the save file
    savefilename = 'spiketrains.pkl'
    cPickle.dump({'train':[], 'test':[]}, savefilename)
    # record all neurons
    config['network']['record_AL'] = "PNs, LNs, drivers"
    config['network']['record_MBext'] = "Dec, Inh"

# set up classifier network
bc = netcontrol.BrainController(p, config)
bc.brain.wire_AL_to_MBext()
bc.brain.reinit_random_weights()
build_network_time = time.time()

# train the network
train_time_neuclar = 0. #the total time spent in training code on the neuclar side
train_time_pynn = 0. # time spent on the PyNN side 
timing_dict = {}
for tp in training_patterns:
    bc.learn_pattern(tp, class_ids, timing_dict=timing_dict)
    if save_spiketrains:
        st = bc.get_spikes()
        st['pattern_name'] = tp[0]
        savedict = cPickle.load(savefilename)
        savedict['train'].append(st)
        cPickle.dump(savedict, savefilename)
    train_time_neuclar += timing_dict['total_train'] - timing_dict['run']
    train_time_pynn += timing_dict['run']
train_time = time.time()

# assess test set
test_results = []
test_time_neuclar = 0.
test_time_pynn = 0.
timing_dict = {}
for tp in testing_patterns:
    test_results.append(bc.test_pattern(tp, timing_dict=timing_dict))
    if save_spiketrains:
        st = bc.get_spikes()
        st['pattern_name'] = tp[0]
        savedict = cPickle.load(savefilename)
        savedict['test'].append(st)
        cPickle.dump(savedict, savefilename)
    test_time_neuclar += timing_dict['manage_test']
    test_time_pynn += timing_dict['run']
test_time = time.time()

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
calc_result_time = time.time()

timings = {"setup":setup_time - start_time,
	   "load_data":load_data_time - setup_time,
	   "convert_data": convert_data_time - load_data_time,
	   "build_network": build_network_time - convert_data_time,
	   "train": train_time - build_network_time,
	   "train_neuclar": train_time_neuclar,
	   "train_pynn": train_time_pynn,
	   "test": test_time - train_time,
	   "test_neuclar": test_time_neuclar,
	   "test_pynn": test_time_pynn,
	   "calc_result": calc_result_time - test_time,
	   "total": calc_result_time - start_time}
	


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
	resultlines.append("Timings:\n")
	timing_names = ['setup',
		   'load_data',
		   'convert_data',
		   'build_network',
		   'train',
		   'train_neuclar',
		   'train_pynn',
		   'test',
		   'test_neuclar',
		   'test_pynn',
		   'calc_result',
		   'total']
	for timing in timing_names:
         resultlines.append("{:<15s}: {:.4f} s\n".format(timing, timings[timing]))
	if save_spiketrains:
         resultlines.append("\n\nSpikes have been saved to {}.".format(savefilename))
         resultlines.append("Saving spikes may have caused significant run time overhead.")
	resultlines.append("\n\n")
	resultlines.append("target\tpredicted\n")
	resultlines.append("------\t---------\n")
	for t,pred in zip(testing_labels, test_results):
         resultlines.append("{}\t{}\n".format(t, class_ids[numpy.argmax(pred)]))
	fd.writelines(resultlines)
	fd.close()
				   
