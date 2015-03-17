#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
$Id$

Classes and functions to create the network.
"""
import os
import logging
#logging.basicConfig()
lg = logging.getLogger(os.path.basename(__file__))
lg.setLevel(logging.WARNING)

pathname = os.path.dirname(__file__)

odordatapath = os.path.join(pathname,       # input data
                            os.path.pardir,
                            os.path.pardir,
                            'data',
                            'surrogate_odor_patterns')

import numpy
import pyNN
pynn_version = pyNN.__version__
import pyNN.random

class BeeBrain(object):
    """
    Container class for all circuits the we model from the bee brain.
    """
    def __init__(self, pynn, config):
        self.pynn = pynn
        self.config = config
        self.AL = AntennalLobe(pynn, config)
        mbtype = config['network']['MB_type']
        assert(mbtype=='AL', 
               'mbtype "{}" not supported in this version'.format(mbtype))
        self.MBcalyx = None
        self.MBext = AttractorDecisionCircuit(pynn, config)
        self.pre_dec = None # array of projections targeting the decision layer.
        self.prjs_calyx_MBext = None
        self.prjs_PN_KC = None
        self.prjs_PN_PCT = None

    def wire_AL_to_MBext(self):
        if self.pre_dec != None:
            raise Exception("trying to wire AL with MBext, but there is" +
                        "already something else wired. That something is" +
                        self.pre_dec + " .")
        self.AL.wire_to_MBext(self.MBext)
        # store which projections are presynaptic to the decision circuit
        self.pre_dec = self.AL.connmat_al_mbext

    def randomize_pre_dec_weights(self, seed=123456):
        """
        set weights from AL to MBext to random values in the interval
                        [wmin, wmax]
        """
        lg.debug('randomizin pre-dec weights.')
        wmax_int = self.config['network'].as_int('w_max_predec')
        wmin_int = self.config['network'].as_int('w_min_predec')
        for prj in numpy.ravel(self.pre_dec):
            if prj == 0:
                continue
            seed = seed + 1 # otherwise we get symmetric weights
            randomize_weights(prj, wmin_int, wmax_int, seed)

    def retrieve_spikes(self, not_older_than=None):
        raise(Exception('deprecated. Refactor to use spikemat format.'))
        alspikes = self.AL.retrieve_spikes(not_older_than=not_older_than)
        if not (self.MBcalyx is None):
            mbspikes = self.MBcalyx.retrieve_spikes(not_older_than=not_older_than)
        else:
            mbspikes = numpy.array([], dtype=float)
        decspikes = self.MBext.retrieve_spikes_dict(poplist=['dec_pops'],
                                                not_older_than=not_older_than)
        return alspikes, mbspikes, decspikes

    def get_predec_weightmat(self):
        """
        Return the weights of all connections targeting the decision layer.

        Returns an array of the same dimensionality as the predec- connmat
        (currently only AL to dec connmat is implemented):
        dim1: glom, 2:pn, 3:decpop, 4:decneuron
        """
        weightmat = numpy.zeros(self.pre_dec.shape, dtype=float)
        wmflat = weightmat.flat
        pdflat = self.pre_dec.flat
        for i,prj in enumerate(pdflat):
            if prj == 0:
                wmflat[i] = numpy.nan
            else:
                wmflat[i] = prj.getWeights(gather=False)[0]
        return weightmat


    def set_predec_weightmat(self, weightmat):
        """
        set the weights in the predec connections according to the weightmat.
        """
        wmflat = weightmat.flat
        pdflat = self.pre_dec.flat
        for i,w in enumerate(wmflat):
            if numpy.isnan(w):
                continue
            else:
                pdflat[i].setWeights(w)


    def get_projections_dict(self):
        """
        return a dictionary with projections.
        Keys: names, values: list of projections
        """
        retdict = {}
        retdict['PN_KC'] = self.prjs_PN_KC
        retdict['PN_PCT'] = self.prjs_PN_PCT
        retdict['calyx_MBext'] = self.prjs_calyx_MBext
        retdict['AL_MBext'] = self.AL.connmat_al_mbext
        return retdict

    def reinit_random_weights(self, seed=123456789):
        """
        reinitialize weights PN->KC, PN->PCT, PCT->KC, KC->MBext.
        """
        rng = numpy.random.RandomState(seed=seed)
        self.randomize_pre_dec_weights(seed=rng.random_integers(low=1e7, high=1e8))

class AntennalLobe(object):
    """
    The Antennal Lobe consists of Glomeruli which are wired by lateral
    inhibition.

    Lateral inhibition is not setup automatically, but must be setup by calling
    setup_lateral_inhibition()!
    """
    def __init__(self, pynn, config):
        """
        pynn - the pynn simulator instance
        config - ConfigObj with network config.
        """
        self.pynn = pynn
        self.config = config
        self.num_gloms = config['network'].as_int('num_glomeruli')
        self.num_PNs = self.config['network'].as_int('num_PNs')
        self.num_LNs = self.config['network'].as_int('num_LNs')
        self.num_neurons =  self.num_gloms * (self.num_PNs + self.num_LNs)
        self.num_drivers_per_glom = config['driver_params'].as_int('drivers_per_glom')

        self.gloms = [Glomerulus(pynn, config) for r in range(self.num_gloms)]
        for gi, g in enumerate(self.gloms):
            for pni, pn in enumerate(g.PNs):
                pn.label = 'PN%d_glom%d'%(pni, gi)
            for lni, ln in enumerate(g.LNs):
                ln.label = 'LN%d_glom%d'%(lni, gi)
        self.driving_type = config['driver_params']['driving_type']
        if self.driving_type == 'weight':
            self.pynn.run(1.) # upload input spikes
        elif self.driving_type == 'rate':
            pass
        else:
            raise(Exception('unknown driving type: %s'%self.driving_type))

        self.li_projections = self._setup_li_projections()

    def _setup_li_projections(self):
        """
        create lateral inhibitory projections between glomeruli. Projections are 
        only initialized, no weight is set.
        """
        li_projections = numpy.zeros((self.num_gloms,
                                           self.num_LNs,
                                           self.num_gloms,
                                           self.num_PNs), dtype=object)
        for gsi,gs in enumerate(self.gloms):
            for lni, ln in enumerate(gs.LNs):
                for gti,gt in enumerate(self.gloms):
                    for pni, pn in enumerate(gt.PNs):
                        li_projections[gsi,lni,gti,pni] = self.pynn.Projection(
                            presynaptic_population=ln,
                            postsynaptic_population=pn,
                            target='inhibitory',
                            method=self.pynn.OneToOneConnector(weights=0.0))
        return li_projections

    def setup_lateral_inhibition_from_config(self):
        """
        Sets up lateral inhibitory connections according to config.
        """
        weights_cfg = eval(self.config['network']['w_LI'])
        if type(weights_cfg) == type([]):
            weights = numpy.array(weights_cfg, dtype=float)
        else:
            weights = float(weights_cfg)
        if numpy.sum(weights) > 0.0001:
            self.connect_lateral_inhibition(weights)

    def connect_lateral_inhibition(self, weights):
        """
        Connect the glomeruli by lateral inhibition. Self-inhibition is avoided.

        parameters:
        weights - the weight of the inhibitory synapse. If weight is a single
                    value, assume uniform connectivity and set all weights to 
                    that value. If weight is a matrix, connect only glomeruli i 
                    and j where weight[i,j] > 0; set weight to weight[i,j].
        Returns:
        NxN numpy array representing connection matrix for N glomeruli,
            containing projection objects at appropriate locations and None
            otherwise.

        """
        ng = len(self.gloms)
        if type(weights) == float:
            weights = numpy.ones((ng,ng)) * weights
            if not self.config['network'].as_bool('self_inhibition'):
                for i in range(ng):
                    weights[i,i] = 0.
        lg.debug('Setting AL weight matrix:')
        lg.debug(weights)
        weights = weights.reshape((ng, 1, ng, 1))
        weights = weights.repeat(self.num_LNs, axis=1)
        weights = weights.repeat(self.num_PNs, axis=3)
        self.set_inh_weightmat(weights)
        lg.debug('AL inhibition weights set.')

    def reset_lateral_inhibition(self):
        """
        set all weights of lateral inhibition projections to zero.
        """
        weightmat = numpy.zeros_like(self.li_projections)
        self.set_inh_weightmat(weightmat)

    def set_inh_weightmat(self, weightmat):
        """
        set the inhibitory weight matrix.
        
        Parameters:
        weightmat - 4-dim array with dim1: index of source glom, dim2: source 
                LN, dim3: target glom, dim4: target PN. NaN elements are
                not written but skipped, which allows for a tremendous speedup
                when only few weights should actually be changed.
        """
        wmflat = weightmat.flat
        lipflat = self.li_projections.flat
        for i,w in enumerate(wmflat):
            if numpy.isnan(w):
                # skip nans
                continue
            lipflat[i].setWeights(w)
 
    
    def get_inh_weightmat(self):
        """
        get the inhibitory weight matrix.
        
        Returns an 4-dim array with dim1: index of source glom, dim2: source 
                LN, dim3: target glom, dim4: target PN.
        """
        weightmat = numpy.zeros(self.li_projections.shape, dtype=float)
        lipflat = self.li_projections.flat
        for i,prj in enumerate(lipflat):
            ind = numpy.unravel_index(i, weightmat.shape)
            weightmat[ind] = prj.getWeights(gather=False)[0]
        return weightmat
    
    def get_pndriver_weightmat(self):
        """
        retrieve the weight matrix for drivers to PNs.
        returns 3-dim matrix with 
         dim1: glom index
         dim2: PN index
         dim3: driver index
        """
        wmat = numpy.zeros((self.num_gloms, 
                            self.num_PNs, 
                            self.num_drivers_per_glom))
        for gi,glom in enumerate(self.gloms):
            pndwm = glom.get_driver_weightmat()
            wmat[gi,:,:] = pndwm.T
        return wmat
    
    def set_pndriver_weightmat(self, wmat):
        """
        set the weights between drivers and PNs.
        takes 3-dim matrix with
         dim1: glom index
         dim2: PN index
         dim3: driver index
        """ 
        for gi,glom in enumerate(self.gloms):
            glom.set_driver_weightmat(wmat[gi,:,:].T)

    def get_lndriver_weightmat(self):
        """
        retrieve the weight matrix for drivers to LNs (the PN outputs).
        returns 3-dim matrix with 
         dim1: glom index
         dim2: LN index
         dim3: PN index
        """
        wmat = numpy.zeros((self.num_gloms, 
                            self.num_LNs, 
                            self.num_PNs))
        for gi,glom in enumerate(self.gloms):
            lnpnwm = glom.get_pnln_weightmat()
            wmat[gi,:,:] = lnpnwm.T
        return wmat
    
    def set_lndriver_weightmat(self, wmat):
        """
        set the weights between PNs and LNs.
        takes 3-dim matrix with
         dim1: glom index
         dim2: LN index
         dim3: PN index
        """ 
        for gi,glom in enumerate(self.gloms):
            glom.set_pnln_weightmat(wmat[gi,:,:].T)

    def randomize_pndriver_weights(self, mean, std):
        """
        Replace the driver->pn weights with values drawn from a normal 
        distribution.
        
        parameters: mean and std of normal distribution
        """
        rng = numpy.random.RandomState(seed=1234567)
        weightmat = self.get_pndriver_weightmat()
        new_weightmat = rng.normal(mean, std, weightmat.shape)
        for nwi,nw in enumerate(new_weightmat.flat):
            if nw < 0.:
                new_weightmat.flat[nwi] = 0.
        self.set_pndriver_weightmat(new_weightmat)

    def randomize_lndriver_weights(self, mean, std):
        """
        Replace the pn->ln weights with values drawn from a normal 
        distribution.
        
        parameters: mean and std of normal distribution
        """
        rng = numpy.random.RandomState(seed=1234567)
        weightmat = self.get_lndriver_weightmat()
        new_weightmat = rng.normal(mean, std, weightmat.shape)
        for nwi,nw in enumerate(new_weightmat.flat):
            if nw < 0.:
                new_weightmat.flat[nwi] = 0.
        self.set_lndriver_weightmat(new_weightmat)
                
    def sparsen_predec_dec_connmat(self, connmat, factor):
        """
        Delete bools from the specified matrix such that only "factor"
        of the connections are True, while reducing the connection overlap as 
        much as "reasonably possible".
        
        Suited for predec-dec connection matrices.
        
        * "reasonably possible" means that I couldn't figure out an algorithm 
        that maximizes the angle between the connection vectors.
        
        Parameters:
            factor - the fraction of entries in each incoming connection vector
                to be True. Rounded up if n_incoming * factor is not integer.
        Returns sparsened weightmat.
        """
        # note: not catching the case where less permutations are possible than
        # required to get full independence, i.e. when factor is too small.
        sp = connmat.shape
        num_keeps = int(numpy.ceil(sp[1] * factor))
        vec = numpy.zeros(sp[1], dtype=bool)
        vec[0:num_keeps] = True
        import itertools
        allperm = numpy.array(list(set(itertools.permutations(vec))), 
                              dtype=bool)
        conn = allperm[:sp[3]]
        conn = conn.T
        conn.shape = (1, sp[1], 1, sp[3])
        conn = numpy.repeat(conn, sp[0], 0)
        conn = numpy.repeat(conn, sp[2], 2)
        return conn
        
    def wire_to_MBext(self, MBext):
        """
        All-to-all connection from PNs to each population in the decision layer.
        """
        num_decpops = self.config['network'].as_int('decision_pops')
        num_decpop_neurons = self.config['network'].as_int('num_dec_neurons')
        # 4-dim array to store AL-MB connections
        #1st dim: gloms, 2nd: PNs, 3rd:decpop, 4th:decpop_neurons
        sparsen_factor = self.config['network'].as_float('MB_sparseness')
        conn_template = numpy.ones((self.num_gloms, 
                                    self.num_PNs, 
                                    num_decpops, 
                                    num_decpop_neurons), dtype=bool)
        if sparsen_factor < 1.:
            conn_template = self.sparsen_predec_dec_connmat(conn_template, 
                                                            sparsen_factor)
        self.connmat_al_mbext = numpy.zeros((self.num_gloms,
                                             self.num_PNs,
                                             num_decpops,
                                             num_decpop_neurons), dtype=object)
                                  
        for ig, g in enumerate(self.gloms):
            for idpops, dpops in enumerate(MBext.dec_pops):
                for idneuron, dneuron in enumerate(dpops):
                    for in_al, n_al in enumerate(g.PNs):
                        if conn_template[ig, in_al, idpops, idneuron]:
                            prj = self.pynn.Projection(n_al, dneuron,
                                    method=self.pynn.AllToAllConnector(),
                                    target='excitatory')
                            self.connmat_al_mbext[ig, 
                                                  in_al,
                                                  idpops, 
                                                  idneuron] = prj
        return
    
    def set_pattern(self, pattern):
        """
        Set the input to the desired activity pattern.

        pattern - array with values between 0 and 1 with as many dimensions as
                    glomeruli
        """
        if self.driving_type == 'weight':
            for p,g in zip(pattern, self.gloms):
                g.set_input_weights(p)
                # avoid re-transfering spikes which have not changed anyway:
                self.pynn._inputChanged = False
        elif self.driving_type == 'rate':
            for p,g in zip(pattern, self.gloms):
                g.set_batch_rates(numpy.array([p]),
                    time_per_pattern=self.config['simulation'].as_float('duration'))
        else:
            raise(Exception("Driving type '%s' not supported."%self.driving_type))

    def set_pattern_batch(self, patterns, time_per_pattern):
        """
        Set a batch of patterns to the AL.

        Parameters:
        patterns - NxM array with N patterns for M glomeruli
        time_per_pattern - time for which each pattern should be presented.
        """
        if self.driving_type == 'weight':
            raise(Exception('setting batches of patterns not supported' +
                                               ' for weight-driven glomeruli.'))
        elif self.driving_type == 'rate':
            for i,g in enumerate(self.gloms):
                g.set_batch_rates(patterns[:,i], time_per_pattern)
        else:
            raise(Exception("Driving type '%s' not supported."%self.driving_type))

    def retrieve_last_rates(self, popname):
        """
        retrieve the rate response of the single neurons within each glomerulus.
        
        Parameters:
        popname - LNs or PNs.

        Returns:
        2-d MxN numpy array, M=number of glomeruli, N=number of LNs/PNs.
        """
        dur = self.config['simulation'].as_float('duration')
        spikemat = self.get_spikemat(popname=popname, not_older_than=dur)
        ratemat = numpy.zeros(spikemat.shape, dtype=float)
        for i,s in enumerate(spikemat.flat):
            ratemat.flat[i] = len(s)/dur*1000.
        return ratemat

    def retrieve_last_pattern(self, popname):
        """
        Retrieve the rate response pattern of the last presented pattern as the
        mean rate within each glomerulus.

        Parameters:
        popname - LNs or PNs
        """
        rates = self.retrieve_last_rates(popname)
        pattern = numpy.mean(rates, axis=1)
        return pattern

    def get_spikemat(self, popname, not_older_than=None):
        """
        get spikemat for the denoted population, i.e. one of 'PNs', 'LNs', 
        'drivers'. Drivers means ORNs.

        returns MxN numpy array for M glomeruli and N PNs/LNs, each element
        contains a numpy array with spike times.
        """
        getspikesfunc = "getSpikes(gather=False)"
        getting_nest_drivers = False
        if popname == 'drivers':
            num_neurons = self.num_drivers_per_glom
            # nest does not support getSpikes() for drivers
            if self.pynn.__name__ == 'pyNN.nest':
                getting_nest_drivers = True
                getspikesfunc = "get('spike_times')[0]"
        else:
            num_neurons = eval("self.num_%s"%popname)
        spikemat = numpy.zeros((self.num_gloms, num_neurons), dtype=numpy.object)
        if not(not_older_than is None):
            ct = self.pynn.get_current_time()
            t_min = ct - not_older_than
            filter = True
            lg.debug('reporting only spikes with time > %f'%t_min)
        else:
            filter = False
        lg.info("Getting spikemat for %s pops in AL."%popname)
        for gi,g in enumerate(self.gloms):
            neurons = eval("g.%s"%popname)
            for ni, neuron in enumerate(neurons):
                spikes = eval("neuron.%s"%getspikesfunc)
                #assert(len(numpy.unique(spikes[:,0])) == 1) # only one unit
                if len(spikes) > 0:
                    if not getting_nest_drivers:                        
                        spikes = spikes[:,1]
                    if filter:
                        t_min_ind = numpy.searchsorted(spikes, t_min)
                        spikes = spikes[t_min_ind:]
                spikemat[gi,ni] = spikes
        return spikemat


    def get_neuron_ids(self):
        """
        Return the neuron ids of each glomerulus. For each glomerulus, first the
        PN ids, then the LN ids.
        """
        ids = []
        for glom in self.gloms:
            ids.extend(glom.get_neuron_ids())
        return ids

    def retrieve_driver_spikes(self):
        """
        retrieve spikes from drivers.
        """
        spikes = []
        drivercounter = 0.
        for glom in self.gloms:
            for dr in glom.drivers:
                drspikes = dr.getSpikes()
                drspikes[:,0] = drivercounter
                drivercounter += 1.
                spikes.append(drspikes)
        allspikes_gdf = numpy.concatenate(tuple(spikes), axis=0)
        return allspikes_gdf
    
class Glomerulus():
    """
    A glomerulus consists of input sources, PNs and LNs.
    """
    def __init__(self, pynn, config):
        """
        create a glomerulus, including stimulus drivers
        """
        self.pynn = pynn
        self.config = config

        try:
            self.min_rate = self.config['driver_params'].as_float('driver_min_rate')
            self.max_rate = self.config['driver_params'].as_float('driver_max_rate')
        except KeyError:
            self.min_rate = 25
            self.max_rate = 55

        #create drivers (ORNs) 
        n_drivers = config['driver_params'].as_int('drivers_per_glom')
        driver_type = config['driver_params']['driver_type']
        self.driving_type = config['driver_params']['driving_type']
        duration = config['simulation'].as_float('duration')
        if driver_type == 'gamma':
            self._gamma_shape = config['driver_params'].as_float('driver_gamma_shape')
        elif driver_type == 'poisson':
            self._gamma_shape = 1.
        self.drivers = self._create_drivers(n_drivers, duration)

        # create PNs
        n_pn = config['network'].as_int('num_PNs')
        self.n_pn = n_pn # avoiding potential confusion when porting to populations in the future
        pn_params = dict([(k,float(v))
                        for k,v in config['PN_params'].dict().items()])
        self.PNs = self._create_neuron_group(n_pn, pn_params)

        # create LNs
        n_ln = config['network'].as_int('num_LNs')
        self.n_ln = n_ln
        ln_params = dict([(k,float(v))
                        for k,v in config['LN_params'].dict().items()])
        self.LNs = self._create_neuron_group(n_ln, ln_params)

        self.sparseconnseed = 1231231 # initial seed for sparsening the connmats

        self.driver_projmat = self._connect_drivers(self.drivers,
#                                                  LNs=self.LNs,
                                                  PNs=self.PNs)
        if self.driving_type == 'rate':
            weight = config['network'].as_float('w_driver_PN')
            wmat = self.get_driver_weightmat()
            for di in range(wmat.shape[0]):
                for pi in range(wmat.shape[1]):
                    if numpy.isnan(wmat[di,pi]):
                        continue
                    else:
                        wmat[di][pi] = weight                                        
#            numpy.zeros((self.driver_projmat.shape), dtype=float)
#            weightmat.fill(weight)
            self.set_driver_weightmat(wmat)            
#            for prj in self.driver_projmat.flat:
#                prj.setWeights(weight)

        self.pre_ln_projs, self.pnln_projmat = \
                                self._connect_pnln(LNs=self.LNs, PNs=self.PNs)
        weight = config['network'].as_float('w_PN_LN')
        for prj in self.pre_ln_projs:
            prj.setWeights(weight)

        recordlist = config['network'].as_list('record_AL')
        if recordlist == ['']:
            lg.info('Recording nothing in AL.')
        else:
            lg.info('recording %s.'%' and '.join(recordlist))
            for pop in recordlist:
                eval("[pop.record(to_file=False) for pop in self.%s]"%pop)
        self.pn_to_pn_prjs = [] # PN to PN projections from calibration

    def _make_spiketrain(self, rates, times, t_stop):
        """
        Create the spiketrain. Uses gamma process. If driver is poisson, gamma
        is set to 1.

        Parameters:
        rates - array of rate values
        times - array of time points at which to change the rate according to
                    the values in rates.
        t_stop - maximum time for spike train.
        """
        spike_times = []
        times = numpy.append(times, t_stop)
        for i,r in enumerate(rates):
            # rate = 1/(scale*shape)
            # scale = 1/(shape*rate)
            gamma_scale = 1/(r * self._gamma_shape)
            dur = times[i+1] - times[i]
            scalefac = 1.3
            while(True):
                st = numpy.cumsum(numpy.random.gamma(shape=self._gamma_shape,
                                               scale=gamma_scale,
                                               size=r*dur*scalefac/1000.))
                # ISI zero must be generated with gamma+1 (Nawrot Rückert 08)
                st = numpy.concatenate((numpy.array([0.]),st))
                st_1st = numpy.random.gamma(shape=self._gamma_shape+1,
                            scale=1./(r * (self._gamma_shape+1)),
                            size=1)
                st += st_1st[0]

                lastidx = numpy.searchsorted(st, dur/1000.)
                lg.debug('make_spiketrain_gamma: len(st)=%d, len(stc)=%d'%(len(st), lastidx))
                stc = st[:lastidx]
                if len(stc) < len(st): # did we clip something?
                    break
                else:
                    lg.info('make_spiketrain_gamma: not enough spikes, redoing with more spikes')
                    scalefac *= 1.1
            stc = stc*1000. # convert to ms
            spike_times.append(stc+times[i])
        st_ret = numpy.concatenate(tuple(spike_times))
        return st_ret

    def _create_drivers(self, num_drivers, duration):
        """
        create spike sources for this glomerulus.
        Parameters:
        num_drivers: number of Drivers
        """
        drivers = [self.pynn.Population(1,
                                    cellclass=self.pynn.SpikeSourceArray)
                                                for d in range(num_drivers)]
        return drivers

    def _connect_drivers(self, drivers, PNs):
        """
        Connect the drivers to the PNs. Returns the connections.
        This does not set a weight.
        
        Returns an MxN array with M the number of drivers, N the number of PNs. 
        """
        connmat = numpy.ones((len(drivers), len(PNs)), dtype=bool)
        sparsen_factor = self.config['network'].as_float('AL_sparseness')
        if sparsen_factor < 0.99:
            connmat_template = self.sparsen_AL_connmat(connmat, 
                                                         sparsen_factor)
        else:
            connmat_template = connmat
        projmat = numpy.zeros((len(drivers), len(PNs)), dtype=object)
        for pi,pn in enumerate(PNs):
            for di,driver in enumerate(drivers):
                if connmat_template[di,pi]:
                    proj = self.pynn.Projection(
                                        presynaptic_population=driver,
                                        postsynaptic_population=pn,
                                        target='excitatory',
                                        method=self.pynn.AllToAllConnector())
                    projmat[di,pi] = proj
        return projmat

    def _connect_pnln(self, LNs, PNs):
        """
        Connect PNs to LNs. Returns the connections. This does not set a weight.
        """
        connmat = numpy.ones((len(PNs), len(LNs)), dtype=bool)
        sparsen_factor = self.config['network'].as_float('AL_sparseness')
        if sparsen_factor < 0.99:
            connmat_template = self.sparsen_AL_connmat(connmat, 
                                                         sparsen_factor)
        else:
            connmat_template = connmat
        projmat = numpy.zeros((len(PNs), len(LNs)), dtype=object)
        projections = []
        for pni,pn in enumerate(PNs):
            for lni,ln in enumerate(LNs):
                if connmat_template[pni,lni]:
                    proj = self.pynn.Projection(presynaptic_population=pn,
                                            postsynaptic_population=ln,
                                            target='excitatory',
                                            method=self.pynn.AllToAllConnector())
                    projections.append(proj)
                    projmat[pni, lni] = proj
        return projections, projmat

    def _create_neuron_group(self, num_neurons, neuron_params):
        """
        create a group of identical neurons and return them as a list of
        pynn.Populations.
        """
        return [self.pynn.Population(1,
                                cellclass=self.pynn.IF_facets_hardware1,
                                cellparams=neuron_params)
                                                    for n in range(num_neurons)]


    def sparsen_AL_connmat(self, connmat, factor):
        """
        Sparsen the connection matrix such that only "factor" of the
        connections are made, while reducing the overlap of incoming between 
        connections for target neurons as much as "reasonably possible".
        
        
        * "reasonably possible" means that I couldn't figure out an algorithm 
        that maximizes the angle between the weight vectors.
        
        Parameters:
            factor - the fraction of connections to be retained per input 
                neuron. Rounded up if n_input * factor is not integer.
        Returns sparsened connection matrix.
        """
        # note: not catching the case where less permutations are possible than
        # required to get full independence, i.e. when factor is too small.
        sp = connmat.shape
        num_keeps = int(sp[0] * factor)
        vec = numpy.zeros(sp[0], dtype=bool)
        vec[0:num_keeps] = True
        import itertools
        allperm = numpy.array(list(set(itertools.permutations(vec))))
        rng = numpy.random.RandomState(seed=self.sparseconnseed)
        self.sparseconnseed += 1
        allperm = rng.permutation(allperm)
        if len(allperm) < sp[1]:
            raise(
            Exception('Not enough permutations with factor %.2f. '%factor +
            "Use lower factor or fix the code to repeat parts of the array."))
        conn = allperm[:sp[1]]
        conn = conn.T
        return conn
    
    def project_LNs_to_glom(self, target_glom, weight):
        """
        Project LN synapses (inhibitory)  onto another glomeruli.

        returns a list of pyNN.Projections

        Parameters:
        target_glom - target glomerulus
        weight - weight to use for connection
        """
        raise(Exception("deprecated - now using inh_weightmat"))
        projections = []
        for ln in self.LNs:
            for pn in target_glom.PNs:
                proj = self.pynn.Projection(presynaptic_population=ln,
                                            postsynaptic_population=pn,
                                            target='inhibitory',
                                            method=self.pynn.AllToAllConnector())
                proj.setWeights(weight)
                projections.append(proj)
        return projections

    def set_driver_weightmat(self, wmat):
        """
        sets the weights of the drivers to PNs in this glomerulus according to 
        the wmat 
         dim1: driver index
         dim2: pn index
        """
        for di in range(wmat.shape[0]):
            for pi in range(wmat.shape[1]):
                if numpy.isnan(wmat[di,pi]) or (
                                            self.driver_projmat[di,pi] == 0):
                    if not (numpy.isnan(wmat[di,pi]) and (
                                            self.driver_projmat[di,pi] == 0)):
                        lg.warn('set_driver_weightmat:nans and zero connections inconsistent.')
                else:
                    self.driver_projmat[di,pi].setWeights(wmat[di,pi])
    
    def get_driver_weightmat(self):
        """
        Gets the weights of the drivers to PNs in this glomerulus. 
        Returns the wmat 
         dim1: driver index
         dim2: pn index
        """
        wmat = numpy.zeros((len(self.drivers), self.n_pn))
        for di in range(wmat.shape[0]):
            for pi in range(wmat.shape[1]):
                proj = self.driver_projmat[di,pi]
                if proj == 0:
                    wmat[di,pi] = numpy.nan
                else:
                    wmat[di,pi] = proj.getWeights(gather=False)[0]
        return wmat

    def set_pnln_weightmat(self, wmat):
        """
        sets the weights of PNs to LNs in this glomerulus according to 
        the wmat 
         dim1: PN index
         dim2: LN index
        """
        for di in range(wmat.shape[0]):
            for pi in range(wmat.shape[1]):
                if numpy.isnan(wmat[di,pi]) or (self.pnln_projmat[di,pi] == 0):
                    if not (numpy.isnan(wmat[di,pi]) and (
                                            self.pnln_projmat[di,pi] == 0)):
                        lg.warn('set_pnln_weightmat:nans and zero connections inconsistent.')
                else:
                    self.pnln_projmat[di,pi].setWeights(wmat[di,pi])
    
    def get_pnln_weightmat(self):
        """
        Gets the weights of the PNs to LNs in this glomerulus. 
        Returns the wmat 
         dim1: PN index
         dim2: LN index
        """
        wmat = numpy.zeros((self.n_pn, self.n_ln))
        for di in range(wmat.shape[0]):
            for pi in range(wmat.shape[1]):
                proj = self.pnln_projmat[di,pi]
                if proj == 0:
                    wmat[di,pi] = numpy.nan
                else:
                    wmat[di,pi] = proj.getWeights(gather=False)[0]
        return wmat

    def set_input_rates(self, p):
        """
        Set the rate of the input driver according to p.

        p should be a value between 0. and 1.0. It is mapped onto the range of
        useful rates.
        """
        raise(Exception('deprecated - use set_batch_rates.'))
        rate_min = self.min_rate
        rate_max = self.max_rate

        rate = rate_min + p * (rate_max - rate_min)
        driver_type = self.config['driver_params']['driver_type']
        if driver_type != 'poisson':
            raise(Exception('rate-based stimulation not supported for driver %s'%driver_type))

        lg.debug("setting rate %f to drivers."%rate)
        for d in self.drivers:
            d.set("rate", rate)

    def set_batch_rates(self, ps, time_per_pattern):
        """
        Set the rate of the input driver according to p.

        p should be a list of values between 0. and 1.0. It is mapped onto the
        range of useful rates. If p contains more than one values, spike trains
        are generated with varying rates, where each spike rate is maintained
        for time_per_pattern ms.

        Parameters:
        ps - list of rates.
        """
        lg.debug("setting batch rates to drivers.")
        rate_min = self.min_rate
        rate_max = self.max_rate
        rates = ps * (rate_max - rate_min) + rate_min
        if self.pynn.__package__ == "pyNN.hardware":
            t_start = 0.
        else:
            t_start = self.pynn.get_current_time()
        times = numpy.arange(len(ps)) * time_per_pattern + t_start
        t_stop = len(ps) * time_per_pattern + t_start
        for d in self.drivers:
            train = self._make_spiketrain(rates, times, t_stop)
            d.set({'spike_times': train})
    
    def get_pre_ln_projs(self):
        """
        Return the projections which drive the LNs.
        """
        import warnings
        warnings.warn('call to deprecated get_pre_ln_projs in Glomerulus',
                      category=DeprecationWarning)
        return self.pre_ln_projs

    def get_neuron_ids(self):
        """
        Return the ids of the neurons in this glomerulus. First PNs, then LNs.
        """
        ids = [netutil.get_population_ids(pn)[0] for pn in self.PNs]
        ids.extend([netutil.get_population_ids(ln)[0] for ln in self.LNs])
        return ids            

class DecisionCircuit(object):
    """
    in its simplest form, only two populations that signal two choices.
    """
    def __init__(self, pynn, config):
        """
        initialize the DecisionCircuit. Create the populations,
        """
        self.config = config
        self.pynn = pynn
        self.dec_pops = []
        self.num_pops = config['network'].as_int('decision_pops')
        self.num_neurons = config['network'].as_int('num_dec_neurons')
        neuron_params = dict([(k,float(v))
                            for k,v in config['DecN_params'].dict().items()])
        for np in range(self.num_pops):
            pop = []
            for nn in range(self.num_neurons):
                pop.append(self.pynn.Population(1,
                                    cellclass=self.pynn.IF_facets_hardware1,
                                    cellparams=neuron_params,
                                    label='decpop_%d_neuron_%d'%(np,nn)))
            self.dec_pops.append(pop)
        if 'Dec' in config['network'].as_list('record_MBext'):
            lg.info('recording from decision neurons.')
            for p in self.dec_pops:
                for pn in p:
                    pn.record(to_file=False)

    def get_decpops(self):
        """
        return list of lists with decision neurons. First list level: decision
        pops, second level: decision neurons.
        """
        raise(Exception("deprecated"))
        return self.dec_pops

    def get_spikemat(self, not_older_than=None):
        """
        returns an MxN array, for M decpops and N decneurons. Each element is
        itself an array of spiketimes.
        """
        if not(not_older_than is None):
            ct = self.pynn.get_current_time()
            t_min = ct - not_older_than
            filter = True
            lg.debug('reporting only spikes with time > %f'%t_min)
        else:
            filter = False
        spikemat = numpy.zeros((self.num_pops, self.num_neurons), dtype=object)
        for dpi, dp in enumerate(self.dec_pops):
            for ni, n in enumerate(dp):
                spikes = n.getSpikes(gather=False)
                #assert(len(numpy.unique(spikes[:,0])) == 1) # only one unit
                if len(spikes) > 0:
                    spikes = spikes[:,1]
                    if filter:
                        t_min_ind = numpy.searchsorted(spikes, t_min)
                        spikes = spikes[t_min_ind:]
                spikemat[dpi,ni] = spikes
        return spikemat

    def get_spikecountmat(self, not_older_than=False):
        """
        Returns an MxN array of spike counts, for M decpops and N
        decneurons.
        """
        spikemat = self.get_spikemat(not_older_than=not_older_than)
        countmat = numpy.zeros(spikemat.shape, dtype=float)
        for i,s in enumerate(spikemat.flat):
            countmat.flat[i] = len(s)
        return countmat

class AttractorDecisionCircuit(DecisionCircuit):
    """
    Decision circuit with an additional inhibitory population giving negative 
    feedback to the decision populations.
    """
    def __init__(self, pynn, config):
        super(AttractorDecisionCircuit, self).__init__(pynn, config)
        self.inh_pops = []
        num_inh_per_pop = config['network'].as_int('num_inh_dec_neurons')
        neuron_params = dict([(k,float(v))
                            for k,v in config['DecN_params'].dict().items()])
                
        for np in range(self.num_pops):
            pop = []
            for nn in range(num_inh_per_pop):
                pop.append(self.pynn.Population(1,
                                    cellclass=self.pynn.IF_facets_hardware1,
                                    cellparams=neuron_params,
                                    label='decpop_inh_%d_neuron_%d'%(np,nn)))
            self.inh_pops.append(pop)
        if 'Inh' in config['network'].as_list('record_MBext'):
            lg.info('recording from decision neurons.')
            for p in self.inh_pops:
                for pn in p:
                    pn.record(to_file=False)
        
        # project each dec pop onto one inh pop
        self.prj_dec_inh = []        
        sparsen_factor = self.config['network'].as_float('MB_sparseness')
        connmat = numpy.ones((self.num_neurons,
                                    num_inh_per_pop), dtype=bool)

        w = config['network'].as_float('w_MBext_dec_inh')
        for dp,ip in zip(self.dec_pops, self.inh_pops):
            #regenerate connectivity matrix each time for added randomness.
            if sparsen_factor < 0.99:
                conn_template = self.sparsen_dec_inh_connmat(connmat, 
                                                            sparsen_factor)
            else:
                conn_template = connmat
            for di,d in enumerate(dp):
                for ii,i in enumerate(ip):
                    if conn_template[di,ii]:
                        prj = pynn.Projection(d, 
                                              i, 
                                              method=pynn.AllToAllConnector(),
                                              target='excitatory')
                        prj.setWeights(w)
                        self.prj_dec_inh.append(prj)

        # project each inh onto all dec
        self.prj_inh_dec = []
        w = config['network'].as_float('w_MBext_inh_dec')
        for i,inh in enumerate(self.inh_pops):
            for d,dec in enumerate(self.dec_pops):
                if i==d:
                    continue # no self-inhibition!
                for di in inh:
                    for dn in dec:
                        prj = pynn.Projection(di, dn, method=pynn.AllToAllConnector(),
                                        target='inhibitory')
                        prj.setWeights(w)
                        self.prj_inh_dec.append(prj)

    def sparsen_dec_inh_connmat(self, connmat, factor):
        """
        Provide a connection matrix for dec_exc to inh neurons with to 
        appropriate connection density to implement sparsen_factor.
        """
        sp = connmat.shape
        num_keeps = int(numpy.ceil(sp[0] * factor))
        vec = numpy.zeros(sp[0], dtype=bool)
        vec[0:num_keeps] = True
        import itertools
        allperm = numpy.array(list(set(itertools.permutations(vec))), 
                              dtype=bool)
        allperm = numpy.random.permutation(allperm)
        conn = allperm[:sp[1]]
        return conn
        
    def get_dec_inh_weights(self):
        weights = [prj.getWeights()[0] for prj in self.prj_dec_inh]
        weight = numpy.unique(weights)
        assert len(weight) == 1
        return weight[0]
        
    def set_dec_inh_weights(self, weight):
        for prj in self.prj_dec_inh:
            prj.setWeights(weight)
            
    def get_inh_dec_weights(self):
        weights = [prj.getWeights()[0] for prj in self.prj_inh_dec]
        weight = numpy.unique(weights)
        assert len(weight) == 1
        return weight[0]
        
    def set_inh_dec_weights(self, weight):
        for prj in self.prj_inh_dec:
            prj.setWeights(weight)
        
    def get_spikemat(self, not_older_than=None, pop='exc'):
        """
        returns an MxN array, for M decpops and N decneurons. Each element is
        itself an array of spiketimes.
        
        parameters:
            not_older_than - how far to look in the past for considering spikes
            pop - either "exc" or "inh"
        """
        # TODO: enable retrieving inhibitory neurons
        if not(not_older_than is None):
            ct = self.pynn.get_current_time()
            t_min = ct - not_older_than
            filter = True
            lg.debug('reporting only spikes with time > %f'%t_min)
        else:
            filter = False
        if pop == 'exc':
            neuroncount = self.num_neurons
            spikepops = self.dec_pops
        elif pop == 'inh':
            neuroncount = self.config['network'].as_int('num_inh_dec_neurons')
            spikepops = self.inh_pops
        spikemat = numpy.zeros((len(spikepops), neuroncount), dtype=object)
        lg.info('retrieving spikemat for %s pop in AttractorDecisionCircuit'%(
                                                            pop))
        for dpi, dp in enumerate(spikepops):
            for ni, n in enumerate(dp):
                spikes = n.getSpikes(gather=False)
                #assert(len(numpy.unique(spikes[:,0])) == 1) # only one unit
                if len(spikes) > 0:
                    spikes = spikes[:,1]
                    if filter:
                        t_min_ind = numpy.searchsorted(spikes, t_min)
                        spikes = spikes[t_min_ind:]
                spikemat[dpi,ni] = spikes
        return spikemat


def connect_fixed_number_pre(pynn, pre_pops, post_pop, number_pre, target, seed=1234567):
    """
    Connects a fixed number of presynaptic populations to each neuron in the
    postsynaptic population.

    Distinguishes between PyNN versions (FromListConnector differs between both).

    Returns a list of Projections

    Parameters:
    pynn - the pynn instance
    pre_pops - list of presyn populations
    post_pop - postsynaptic population
    number_pre - number of presyn populations to connect to every neuron in the
                postsyn population.
    seed - seed for the random number generator which picks the presyn neurons.
    """
    def _make_tuple_06(pre_pop, pre_index, post_pop, post_index, weight=0.0,
                        delay=0.1):
        """
        returns connection tuple (pre_addr, post_addr, weight, delay)
        with weight and delay
        """
        pre_addresses = [pg for pg in pre_pop.addresses()]
        pre_addr = pre_addresses[pre_index]
        post_addresses = [pg for pg in post_pop.addresses()]
        post_addr = post_addresses[post_index]
        return (pre_addr, post_addr, weight, delay)

    def _make_tuple_07plus(pre_pop, pre_index, post_pop, post_index, weight=0.0,
                        delay=0.1):
        """
        returns connection tuple (pre_addr, post_addr, weight, delay)
        with weight and delay
        """
        return (pre_index, post_index, weight, delay)

    if pynn_version.split(' ')[0] == "0.6.0":
        make_tuple = _make_tuple_06
    else:
        make_tuple = _make_tuple_07plus
    connection_dicts = []
    rng = numpy.random.RandomState(seed=seed)
    for post_idx in range(len(post_pop)):
        prepop_indices = rng.permutation(len(pre_pops))[:number_pre]
        for idx in prepop_indices:
            pre_pop = pre_pops[idx]
            tuple = make_tuple(pre_pop, 0, post_pop, post_idx)
            connection_dicts.append({'prepop':pre_pop, 'tuple':tuple})
    ret_prjs = []
    for pp in pre_pops:
        tuple_list = [t['tuple'] for t in connection_dicts if t['prepop'] == pp]
        if len(tuple_list) == 0:
            lg.warn("%s has no postsyn partner in %s"%(pp.label, post_pop.label))
            continue
        ret_prjs.append(pynn.Projection(pp, post_pop,
                method=pynn.FromListConnector(conn_list=tuple_list),
                target=target))
    return ret_prjs


class HardwareWeightRNG(pyNN.random.NumpyRNG):
    def __init__(self, seed, parallel_safe=True):
        pyNN.random.NumpyRNG.__init__(self, 
                                      seed=seed,
                                      parallel_safe=parallel_safe)
                                                
    def _next(self, distribution, n, parameters):
        if distribution != 'random_integers':
            raise(Exception('Distribution is not random_integers - this should not happen.'))
        wmin = parameters['wmin']
        wmax = parameters['wmax']
        type = parameters['type']
        if type == 'excitatory':
            abs_wmax = 0.005
        elif type == 'inhibitory':
            abs_wmax = 0.015
        else:
            raise(Exception('type "%s" not understood.'))
        ti = self.rng.random_integers(low=wmin, high=wmax, size=n)
        ti = ti * abs_wmax/15.
        return ti

def randomize_weights(prj, wmin_int, wmax_int, seed=1234567):
    """
    Custom randomize_weights method taking into account the hardware-specific
    weight constrains: min weights, max weights, 4 bit weights.

    Parameters:
    prj - the Projection to modify.
    wmin_int, wmax_int - the range in which to modify the weights. 0, 15 will
                        cover the entire 4 bit range.
    """
    type = prj.target
    rd = pyNN.random.RandomDistribution(distribution='random_integers',
                                        rng=HardwareWeightRNG(seed=seed),
                                        parameters={'wmin':wmin_int,
                                                    'wmax':wmax_int,
                                                    'type':type})
    prj.randomizeWeights(rand_distr=rd)
