# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:06:52 2013

@author: micha
"""

def _get_connection_tuples06(prj):
    return [ct for ct in prj.connections()]

def _get_connection_tuples07plus(prj):
    conn_tuples = [(c.source,c.target) for c in prj.connections]
    return conn_tuples

def _get_population_ids06(pop):
    return [id for id in pop.ids()]

def _get_population_ids07plus(pop):
    return [id for id in pop]

import pyNN
pynn_version = pyNN.__version__

class PynnVersionWorkaround(object):
    """
    Decorator to provide version-dependent functions to work around
    incompatibilities between different pynn versions.
    """
    def __init__(self, f):
        if f.__name__ == 'get_connection_tuples':
            if pynn_version.split(' ')[0] == "0.6.0":
                self.f = _get_connection_tuples06
            else:
                self.f = _get_connection_tuples07plus

        elif f.__name__ == 'get_population_ids':
            if pynn_version.split(' ')[0] == "0.6.0":
                self.f = _get_population_ids06
            else:
                self.f = _get_population_ids07plus

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

@PynnVersionWorkaround
def get_connection_tuples(prj):
    pass

@PynnVersionWorkaround
def get_population_ids(pop):
    pass

        

