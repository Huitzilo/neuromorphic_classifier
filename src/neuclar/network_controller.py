#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
$Id$

Classes and fuinctions to control network and classifier operation.
"""

# just run the network.

import os
import sys
import time
import csv
import cPickle
import logging
lg = logging.getLogger(os.path.basename(__file__))
lg.setLevel(logging.INFO) # set to level 5 (< logging.DEBUG) to obtain pre and post patterns.

import configobj
import numpy

#import NeuroTools.signals as nts

from network import AntennalLobe
from network import BeeBrain
#from utils import utility_funcs
#import init_sim

def usage():
    print "Usage: %s configfile sim_type"%os.path.basename(__file__)
    print "configfile: Path to network configuration file"
    print "sim_type: sim or hw"

from neuclar.network_utilities import *

class ALController(object):
    """
    Setup the antennal lobe network and control simulation, data presentation etc.
    """
    def __init__(self, pynn, config):
        lg.info('building network')
        self.pynn = pynn
        self.net = AntennalLobe(pynn, config)

    def run_network(self, duration):
        lg.info('starting simulation for %.1f ms'%duration)
        self.pynn.run(duration)

    def set_pattern(self, pattern):
        lg.info("setting pattern in the AL.")
        self.net.set_pattern(pattern)

    def set_pattern_batch(self, patterns, time_per_pattern):
        """
        set the driver spike rates such that all patterns are presented in
        sequence.
        """
        lg.info('setting pattern batch in the AL.')
        self.net.set_batch_pattern(patterns, time_per_pattern)

    def retrieve_spikes(self):
        """
        Retrieve the spikes produced in the network.
        """
        lg.info('retrieving spikes.')
        return self.net.retrieve_spikes()        

class BrainController(object):
    """
    set up the honeybee brain, control simulation, present stimuli, learn.
    """
    def __init__(self, pynn, config):
        lg.info('setting up bee brain.')
        self.pynn = pynn
        self.config = config
        self.stim = 0
        self.brain = BeeBrain(pynn, config)
        assert (config['simulation']['calib_AL'] == 'False'),\
               "This version does not support AL calibration."
        if self.config['network'].has_key('randomize_weights_pndriver'):
            # driver -> PNs
            std = config['network'].as_float('randomize_weights_pndriver')
            if std > 0.:
                orig_weight = self.config['network'].as_float('w_driver_PN')
                self.brain.AL.randomize_pndriver_weights(orig_weight, std)
        if self.config['network'].has_key('randomize_weights_lndriver'):
            # PNs -> LNs
            std = config['network'].as_float('randomize_weights_lndriver')
            if std > 0.:
                orig_weight = config['network'].as_float('w_PN_LN')
                self.brain.AL.randomize_lndriver_weights(orig_weight, std)
        self.brain.AL.setup_lateral_inhibition_from_config()

    def set_pattern(self, pattern):
        """
        set the given input pattern as stimulus in the AL.
        """
        lg.info('setting stimulation pattern in the AL.')
        self.brain.AL.set_pattern(pattern)

    def set_pattern_batch(self, patterns, time_per_pattern):
        """
        set the given input pattern as stimulus in the AL.
        """
        lg.info('setting stimulation pattern in the AL.')
        self.brain.AL.set_pattern_batch(patterns, time_per_pattern)

    def run_network(self, duration):
        """
        run the network for duration ms.
        """
        lg.info('running the simulation for %.1f ms.'%duration)
        self.pynn.run(duration)
        lg.info('run completed.')
        
    def get_spikes(self, duration=None):
        """
        Retrieve spikes from all neurons after one stimulus.
        parameters: 
            duration - how far to look back for recorded spikes. If None, guess
                from config. 
        Returns dictionary with spike matrices.
        """
        if duration is None:
            duration = self.config['simulation'].as_float('duration')
        # get ORN spikes
        driverspikes = self.brain.AL.get_spikemat('drivers', 
                                                  not_older_than=duration)
        # get PN spikes
        pnspikes = self.brain.AL.get_spikemat('PNs', not_older_than=duration)
        # get LN spikes
        lnspikes = self.brain.AL.get_spikemat('LNs', not_older_than=duration)
        # get MB dec exc spikes 
        decexcspikes = self.brain.MBext.get_spikemat(pop='exc', 
                                                     not_older_than=duration)
        # get MB dec inh spikes 
        decinhspikes = self.brain.MBext.get_spikemat(pop='inh', 
                                                     not_older_than=duration)
        ret = {'drivers':driverspikes,
               'PNs': pnspikes,
               'LNs': lnspikes,
               'dec_exc': decexcspikes,
               'dec_inh': decinhspikes}
        return ret

    def test_pattern(self, pattern_tuple, class_ids='not used', timing_dict=None):
        """
        Present the pattern and determine the network's choice.

        Returns the number of spikes produced in each decision population.

        Parameters:
        pattern_tuple - pattern tuple as returned from PatternServer
                            (id, pattern, classlabel)
        class_ids - list of strings containing all possible class labels (not 
                        used but necessary in classifiers)
        timing_dict - dictionary in which times for 'run' and 'manage' will be
                    stored (for benchamrking).
        """
        start_time = time.time()
        lg.info('testing pattern.')
        self.stim +=1
        id = pattern_tuple[0]
        pattern = pattern_tuple[1]
        target = pattern_tuple[2]
        self.set_pattern(pattern)
        pat_creat_time = time.time()
        duration = self.config['simulation'].as_float('duration')
        self.run_network(duration)
        post_run_time = time.time()
        if self.pynn.__package__ == 'pyNN.hardware':
            t_back = None
        else:
            t_back = duration
        dn_spikecounts = self.brain.MBext.get_spikecountmat(
                                                        not_older_than=t_back)
        dec_pop_rates = numpy.mean(dn_spikecounts, axis=1)
        lg.info('pattern %s %s %s yielded response %s'%(id, str(pattern),
                                                target, str(dec_pop_rates)))        
        end_time = time.time()
        if not (timing_dict is None):
            timing_dict['total_test'] = end_time - start_time
            timing_dict['create_spiketrains'] = pat_creat_time - start_time
            timing_dict['run'] = post_run_time - pat_creat_time
            timing_dict['compute_rates'] = end_time - post_run_time
        return dec_pop_rates
    
        
    def learn_pattern(self, pattern_tuple, class_ids, timing_dict=None):
        """
        Present a pattern and update the weights in the network according to the
        Fusi learning rule.

        Returns a boolean value indicating whether the classification of the
        pattern was correct when it was initially presented, or None when there
        was no classifier output.

        Parameters:
        pattern_tuple - pattern tuple as returned from PatternServer
                            (id, pattern, classlabel)
        class_ids - list of strings containing all possible class labels
        timing_dict - dictionary in which times for 'run' and 'manage' will be
                    stored (for benchmarking).

        """
        start_time = time.time()
        lg.info('performing learning.')
        # determine winner population and class
        dec_pop_rates = self.test_pattern(pattern_tuple, timing_dict=timing_dict)
        post_test_time = time.time()
        id = pattern_tuple[0]
        target = pattern_tuple[2]
        # determine if classification is correct
        winner = numpy.argmax(dec_pop_rates)
        winner_id = class_ids[winner]
        dec_correct = winner_id == pattern_tuple[2]
        lg.info('Classifier: %s %s -> %s -- %s.'%(
                id, target, winner_id, ['WRONG','CORRECT'][int(dec_correct)]))
        lg.debug("dec_pop_rates: %s"%str(dec_pop_rates))
        lg.debug("argmax(dec_pop_rates): %d"%numpy.argmax(dec_pop_rates))
        lg.debug('class_ids: %s'%str(class_ids))
        assess_classification_time = time.time()        
        
        if numpy.sum(dec_pop_rates) > 1.: #there was at least one spike
            # update weights accordingly
            self.change_predec_weights_learning(dec_pop_rates, dec_correct)
            if self.config['learningrule'].as_bool('learn_AL_inh'):
                pnrates = self.brain.AL.retrieve_last_rates('PNs')
                if not numpy.any(pnrates > self.config['learningrule'].as_float('pn_learn_thresh')):
                    # decrease overall inhibition if pn rate is too low for learning
                    self.change_AL_inh_weights_const(-0.015/15)
                else:
                    lnrates = self.brain.AL.retrieve_last_rates('LNs')
                    self.change_AL_inh_weights_learning(pnrates, lnrates, dec_correct)
        else:
            lg.info('Classifier: %s %s -> %s.'%(id, target, 'no output spikes'))
            # increase all weights by one step
            self.change_predec_weights_const(0.005/15.)
        end_time = time.time()
        if not (timing_dict is None):
            timing_dict['total_train'] = end_time - start_time
            timing_dict['assess_classification'] = \
                    assess_classification_time - post_test_time
            timing_dict['compute_new_weights'] = \
                    end_time - assess_classification_time
        return dec_pop_rates

    def change_predec_weights_const(self, dw):
        """
        Change the weight coming into the decision layer by constant amount dw.
        """
        lg.info('increasing all predec weights by %.5f'%dw)
        connmat = self.brain.AL.connmat_al_mbext
        w_min = self.config['learningrule'].as_float('w_min')
        w_max = self.config['learningrule'].as_float('w_max')
        for conn in connmat.flat:
            if conn == 0:
                continue
            w = conn.getWeights(gather=False)[0]
            new_w = w + dw
            if new_w > w_max:
                new_w = w_max
            elif new_w < w_min:
                new_w = w_min
            conn.setWeights(new_w)


    def change_predec_weights_learning(self, dec_pop_rates, dec_correct):
        """
        Modify the weights to the decision population according to the learning
        rule.

        Parameters:
        dec_pop_rates - list of rates from the decision populations
        dec_correct - boolean indicating whether decision was correct
        """
        winner = numpy.argmax(dec_pop_rates)

        # consider only spikes which occurred during last presentation
        rank_thresh = self.config['learningrule'].as_int('rank_thresh')
        rate_thresh = self.config['learningrule'].as_float('rate_thresh')
        duration = self.config['simulation'].as_float('duration')

        #obtain pre spikes
        if self.brain.MBcalyx is None:
            # must be AL then
            lg.debug('Learning from AL.')
            pre_spikes = self.brain.AL.get_spikemat('PNs',
                                                        not_older_than=duration)
        else:
            raise(Exception('need to refactor to spikemat.'))
            # All KCs project to the decision layer. Get spikes for KC population.
            lg.debug('Learning from MB.')
            mb_spike_dict = self.brain.MBcalyx.retrieve_spikes(poplist=['KCs'],
                                                        not_older_than=duration)
            pre_spikes = mb_spike_dict['KCs']

        # calculate pre rates
        pre_rates = numpy.zeros(pre_spikes.shape, dtype=float)
        for i,s in enumerate(pre_spikes.flat):
            pre_rates.flat[i] = len(s)/duration*1000.

        lg.info('Pre pattern: %s'%str(["%.2f"%numpy.mean(s) for s in pre_rates]))
        # find the n highest responding units, n < rank_thresh
        units_sortidx = numpy.argsort(pre_rates.flat)
        units_sortidx = units_sortidx[::-1]
        # check whether rate_thresh or rank_thresh is relevant
        if rank_thresh > (len(units_sortidx)-1):
            lg.debug('setting rank_thresh of %d to max rank of %d'%(
                                            rank_thresh,len(units_sortidx)-1))
            rank_thresh = (len(units_sortidx)-1)
        if pre_rates.flat[units_sortidx[rank_thresh]] < rate_thresh:
            rates = [pr for pr in pre_rates.flat[units_sortidx]]
            cutoff = numpy.searchsorted(pre_rates.flat[units_sortidx[::-1]], rate_thresh)
            cutoff -= len(pre_rates.flat)
            cutoff *= -1
            lg.debug('rate_threshing at rank %d'%cutoff)
        else:
            cutoff = rank_thresh
            lg.debug('rank_threshing at %d'%cutoff)
        unit_id_tup = []
        for idx in units_sortidx[:cutoff]:
            unit_id_tup.append(numpy.unravel_index(idx,pre_rates.shape))
        lg.info('learning: updating weights from %d pre_units'%len(unit_id_tup) +
                " targeting decpop %d"%(winner))
        
        # set compute mode for dw
        w_min = self.config['learningrule'].as_float('w_min')
        w_max = self.config['learningrule'].as_float('w_max')
        try:
            calc_dw = self.config['learningrule']['dw_style']
        except KeyError:
            calc_dw = 'static_dw'
        if calc_dw == 'static_dw':
            delta_w_plus_int = self.config['learningrule'].as_int('delta_w_plus_int')
            delta_w_minus_int = self.config['learningrule'].as_int('delta_w_minus_int')
            delta_w_plus = float(delta_w_plus_int) * 0.005/15.
            delta_w_minus = float(delta_w_minus_int) * 0.005/15.
            if dec_correct:
                dw = delta_w_plus
            else:
                dw = -delta_w_minus
        elif calc_dw == 'soltani_wang':
            if dec_correct:
                w_ref = w_max
            else:
                w_ref = w_min
        # loop over the to-be-modified connections and set new weight.
        connmat = self.brain.AL.connmat_al_mbext
        for id in unit_id_tup:
            conns = connmat[id[0], id[1], winner, :]
            for conn in conns.flat:
                if conn == 0:
                    continue
                w = conn.getWeights(gather=False)[0]
                if calc_dw == 'static_dw':
                    new_w = w + dw
                    if new_w > w_max:
                        new_w = w_max
                    elif new_w < w_min:
                        new_w = w_min
                elif calc_dw == 'soltani_wang':
                    #TODO: compute dw like Soltani/Wang
                    # dw = 1/1+exp(-w_max)
                    raise(Exception(
                    'Still need to figure out how S-W actually computerd dw.'))
                        
                if numpy.abs(new_w - w) < 0.000001:
                    lg.debug('learning: not changing weight ' +
                                        '(old: %.5f, new: %.5f)'%(w, new_w))
                    continue
                else:
                    lg.debug('learning: changing weights for '+
                        'glom: %d pn:%d decp:%d'%(id[0], id[1], winner) +
                                        '(old: %.5f, new: %.5f)'%(w, new_w))
                    conn.setWeights(new_w)
