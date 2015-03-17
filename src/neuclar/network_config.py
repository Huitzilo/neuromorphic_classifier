# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:20:16 2013

@author: micha
"""
import configobj
import StringIO

configstr = StringIO.StringIO("""
[network]
num_glomeruli    = 10         # number of glomeruli
w_driver_PN      = 0.005     # weight from drivers to LNs or PNs.
AL_sparseness    = 1.0    # which proportion of drivers should connect to each PN
w_PN_LN          = 0.004     # weight from PNs to LNs
w_LI             = 0.0000     # weight of inhibition between glomeruli (probably overridden)
num_PNs          = 7          # number of PNs per glomerulus
num_LNs          = 6         # number of LNs per glomerulus
self_inhibition  = False      # whether a glomerulus should also inhibit itself
record_AL        = LNs, PNs   # which populations to record in the AL
MB_type          = AL         # mushroom body type
decision_pops    = 3          # number of populations in the decision circuit
num_dec_neurons  = 8          # number of neurons per decision population
num_inh_dec_neurons  = 8          # number of inhibitory neurons per decision population
MB_sparseness    = 1.0        # sparseness of predec->dec and dec->dec_inh projections
record_MBext     = Dec        # which neurons to record in the MBext circuit
w_min_predec     = 3          # min weight for pre-dec prjs in random init (4-bit int)
w_max_predec     = 10          # max weight for pre-dec prjs in random init (4-bit int)
w_MBext_dec_inh  = 0.0035     # weight from dec to inh in MBext
w_MBext_inh_dec  = 0.0100     # weight from inh to dec in MBext
calib_AL = False
randomize_weights_pndriver = 0. #0.003 # 0.0002 is sigma^2 = 30% at w=0.005 
randomize_weights_lndriver = 0. #0.003 # 0.0002 is sigma^2 = 30% at w=0.005

[simulation]
duration         = 1000. # total simulation duration in ms (also used for drivers)
mappingOffset    = 0    # mapping offset for neuron creation
store_PNs        = True
store_spikes     = False
calib_AL         = False      # calibrate AL
calib_method     = calibrate_AL_weightmat_factor
calib_duration   = 2000.
AL_calib_cachepath = results/calib_weightmats/IrisVrecQuick_station309-10glom.cPickle

[driver_params] # parameters for drivers (spike sources, ORNs)
drivers_per_glom = 6           # number of inputs per glomerulus
driver_type      = gamma     # poisson, gamma
driver_gamma_shape = 5      # gamma shape
driving_type     = rate      # how to drive PNs, either by modifying weight or rate
driver_min_rate  = 25
driver_max_rate  = 55

[classifier]
type = sim
type_kwargs = ()      # will be evaluated to kwargs
dataset = IrisVRecs
classifier_type = NeuralClassifier # NeuralClassifier, NaiveBayes

[IrisVRecs]
ng_num_nodes = 10
distfun = manhattan        # euclidian, manhattan, sigmoid
LI_type = uniform          # NG, corr, rand_uni, uniform
AL_li_weight = 0.015       # inh. weight between two glomeruli if they are connected (scales down in rand & corr)
AL_li_q = 0.5              # overwritten in perf_li script
crossval_folds = 5
repetitions = 1
cv_seed = 1234567
jobrunner = SerialProcessJobRunner
profile_name = 'neurolearn'

[learningrule]
w_max = 0.005           # max. exc. weight on hw
w_min = 0.000333333     # minimum weight
delta_w_plus_int = 5    # one weight step
delta_w_minus_int = 5   # one weight step
rate_thresh = 35        # threshold in spike rate above which to modify synapses
rank_thresh = 10000000  # threshold in firing rate rank until which to modify synapses (ROT:Nal/nclasses)
learn_AL_inh = False    # do al plasticity?
al_inh_max = 0.010      # alplast: max li
al_inh_min = 0.005      # alplast: min li
pn_learn_thresh = 25    # alplast: thresh for pn 
ln_learn_thresh = 25    # alplast: thresh for ln
dw_inh_minus = 0.001    # alplast: dw for weight decrease
dw_inh_plus = 0.001     # alplast: dw for weight increase

[PN_params]
# pyNN: g_leak in nS, cm in nF, # defaults
v_reset     = -80.0     # -80.0
e_rev_I     = -80.0     # -80.0
v_rest      = -65.0     # -65.0
v_thresh    = -55.0     # -55.0
g_leak      =  10.0     #  40.0
tau_syn_E   =  5.0      #  30.0 doesn't have an impact in HW, but is important for compatibility with nest
tau_syn_I   =  5.0      #  30.0

[LN_params]
v_reset     = -80.0     # -80.0
e_rev_I     = -80.0     # -80.0
v_rest      = -65.0     # -65.0
v_thresh    = -55.0     # -55.0
g_leak      =  10.0     #  40.0
tau_syn_E   =  5.0      #  30.0
tau_syn_I   =  5.0      #  30.0

[KC_params]
v_reset     = -80.0     # -80.0
e_rev_I     = -80.0     # -80.0
v_rest      = -65.0     # -65.0
v_thresh    = -55.0     # -55.0
g_leak      =  10.0     #  40.0
tau_syn_E   =  5.0      #  30.0
tau_syn_I   =  5.0      #  30.0

[PCT_params]
v_reset     = -80.0     # -80.0
e_rev_I     = -80.0     # -80.0
v_rest      = -65.0     # -65.0
v_thresh    = -55.0     # -55.0
g_leak      =  10.0     #  40.0
tau_syn_E   =  5.0      #  30.0
tau_syn_I   =  5.0      #  30.0

[DecN_params]
v_reset     = -80.0     # -80.0
e_rev_I     = -80.0     # -80.0
v_rest      = -65.0     # -65.0
v_thresh    = -55.0     # -55.0
g_leak      =  10.0     #  40.0
tau_syn_E   =  5.0      #  30.0
tau_syn_I   =  5.0      #  30.0
""")
software_example_config = configobj.ConfigObj(configstr)

sparsenet_config_str = StringIO.StringIO("""
[network]
AL_sparseness    = 0.5    # which proportion of drivers should connect to each PN
MB_sparseness    = 0.5        # sparseness of predec->dec and dec->dec_inh projections

[driver_params]
driver_min_rate  = 50
driver_max_rate  = 100
""")
software_sparse_config = configobj.ConfigObj(software_example_config.copy())
update = configobj.ConfigObj(sparsenet_config_str)
software_sparse_config.merge(update)

hardware_sparse_config_str = StringIO.StringIO("""
[network]
w_driver_PN      = 0.0025     # weight from drivers to LNs or PNs.
w_PN_LN          = 0.0035     # weight from PNs to LNs
w_LI             = 0.002
AL_sparseness    = 0.5    # which proportion of drivers should connect to each PN
MB_sparseness    = 0.5        # sparseness of predec->dec and dec->dec_inh projections
w_MBext_dec_inh  = 0.0025     # weight from dec to inh in MBext
w_MBext_inh_dec  = 0.0150     # weight from inh to dec in MBext

[driver_params]
driver_min_rate  = 20
driver_max_rate  = 70

[learningrule]
rate_thresh = 25

[classifier]
type = hw

[IrisVRecs]
AL_li_q = 1.
AL_li_weight = 0.002

[MNISTVRecs]
AL_li_q = 1.
AL_li_weight = 0.002

[Ring2DVRecs]
AL_li_q = 1.
AL_li_weight = 0.002

[simulation]
AL_calib_cachepath = results/calib_weightmats/IrisVrecQuick_station309-10glom_sparse.cPickle
""")
hardware_sparse_config = configobj.ConfigObj(software_example_config.copy())
update = configobj.ConfigObj(hardware_sparse_config_str)
hardware_sparse_config.merge(update)

