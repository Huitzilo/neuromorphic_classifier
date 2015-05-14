#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
$Id$

Data samplers sample the data with virtual receptors.
"""

import os
import logging
lg = logging.getLogger(os.path.basename(__file__))
lg.setLevel(logging.INFO)

import numpy
import mdp
datapath = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'data')

mnist_samplepoints = numpy.load(os.path.join(datapath, 'mnist_vrec_pos.npy'))
iris_samplepoints = numpy.load(os.path.join(datapath, 'iris_vrec_pos.npy'))

neural_gas_parameters = {
    'num_nodes': 10,
    'start_poss': None,
    'epsilon_i': 0.3,               # initial epsilon
    'epsilon_f': 0.05,              # final epsilon
    'lambda_i': 30.,                 # initial lambda
    'lambda_f': 0.01,                 # final lambda
    'max_age_i': 20,                # initial edge lifetime
    'max_age_f': 200,               # final edge lifetime
    'max_epochs': 50.,
    'n_epochs_to_train': None,
    'input_dim': None,
    'dtype': None
}

def vrconvert(data, samplepoints):
    vrs = VirtualReceptorSampler()
    vrs.set_samplepoints(samplepoints)
    data_sampled = vrs.sample_data(data)
    return data_sampled

class VirtualReceptorSampler(object):
    """
    Samples the data using virtual receptors.
    """
    def __init__(self):
        self.samplepoints = None
    
    def set_samplepoints(self, samplepoints):
        """
        Set the locations of the virtual receptors.
        Parameters:
        samplepoints - NxM numpy array for N samplepoints having M dimensions.
        """
        self.samplepoints = samplepoints
        
    def sample_data(self, data, dist_type='manhattan'):
        """
        Samples the provided data using the trained sampler.
        Returns an MxO matrix with M data points of dimensionality O, where O is
        equal to the number of sample points.
        """
        if self.samplepoints is None:
            raise(Exception('No samplepoints available.'))
        samplepoints = self.samplepoints
        lg.debug('NG sampler nodepos:')
        lg.debug(str(samplepoints))
        lg.debug('NG sampler data:')
        lg.debug(str(data))
        # calculate distance of data points to samplepoint
        def scale_0_1(resultmat):
            d_min = numpy.min(resultmat, axis=0)
            d_min_mat = numpy.tile(d_min, (len(resultmat), 1))
            d_max = numpy.max(resultmat, axis=0)
            d_max_mat = numpy.tile(d_max, (len(resultmat), 1))
            resultmat = (resultmat - d_min_mat) / (d_max_mat - d_min_mat)
            return resultmat
        def euclidian():
            res = numpy.zeros((data.shape[0], len(samplepoints)))
            for i, sp in enumerate(samplepoints):
                spm = numpy.tile(sp, (len(data), 1))
                dif = spm - data
                dss = numpy.sum(dif**2, axis=1)
                ds = numpy.sqrt(dss)
                res[:,i] = ds
            resmat = 1. - scale_0_1(res)
            return resmat
        def manhattan():
            res = numpy.zeros((data.shape[0], len(samplepoints)))
            for i, sp in enumerate(samplepoints):
                spm = numpy.tile(sp, (len(data), 1))
                dif = spm - data
                dss = numpy.sum(numpy.abs(dif), axis=1)
                res[:,i] = dss
            resmat = 1. - scale_0_1(res)
            return resmat
        def sigmoid():
            res = manhattan(samplepoints)
            exponent = 3 * (res - 1.) # make sparser (-1.) and expand to three stddev (*3)
            res = 1./(1. + numpy.exp(- exponent))
            return scale_0_1(res)
        distfun = eval(dist_type)
        res = distfun()
        # scale each column between 0 and 1, such that the largest distance
        # in each column becomes zero and the smallest distance becomes 1.
        lg.debug('NG sampler result:')
        po = numpy.get_printoptions()
        numpy.set_printoptions(precision=4, threshold=2e9)
        lg.debug(numpy.array_str(res, max_line_width=160, precision=3, suppress_small=True))
        numpy.set_printoptions(**po)
        return res


class NeuralGasSampler(VirtualReceptorSampler):
    """
    Samples the data with a neural gas.
    """
    def train_sampler(self, data, sampler_config):
        """
        Train the sampler to efficiently sample the data.

        Parameters:
        data - the data to sample, NxM array, with N data points of dimension M.
        sampler_config - dictionary with parameters for the neural gas.
        """
        num_nodes = sampler_config['num_nodes']
        # randomly sample num_nodes data points as initial node position
        initial = mdp.numx.take(data,
                                mdp.numx.random.random_integers(high=len(data)-1,
                                                                low=0,
                                                                size=num_nodes),
                                axis=0)
        sampler_config['start_poss'] = initial
        ng = mdp.nodes.NeuralGasNode(**sampler_config)
        ng.train(data)
        self.samplepoints = ng.get_nodes_position()
        self.ng = ng
        self.sampler_config = sampler_config





