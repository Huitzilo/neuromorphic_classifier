# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:02:28 2015

@author: micha
"""

class NeuclarTimings(object):
    """
    class to keep track of the time spent in the various phases of classifier 
    training and testing.
    """
    def __init__(self):
        self.training_times = {'create_spike_trains': 0.,
                               'run': 0.,
                               'identify_winner_pop': 0.,
                               'compute_updated_weights' :0.,
                               'total_train': 0.}
        self.testing_times = {'create_spike_trains': 0.,
                               'run': 0.,
                               'identify_winner_pop': 0.,
                               'total_test': 0.}
                               
                               
    def update_training_times(self, timing_dict):
        self.training_times['create_spiketrains'] += \
                                            timing_dict['create_spiketrains']
        self.training_times['run'] += timing_dict['run']
        self.training_times['identify_winner_pop'] += (
                                        timing_dict['compute_rates'] + 
                                        timing_dict['assess_classification'])
        self.training_times['compute_updated_weights'] += 
                                        timing_dict['compute_new_weights']
        self.training_times['total_train'] += timing_dict['total_train']
    
    def update_testing_times(self, timing_dict):
        self.testing_times['create_spike_trains'] += \
                                            timing_dict['create_spiketrains']
        self.testing_times['run'] += timing_dict['run']       
        self.testing_times['identify_winner_pop'] += \
                                            timing_dict['compute_rates']
        self.testing_times['total_test'] += timing_dict['total_test']