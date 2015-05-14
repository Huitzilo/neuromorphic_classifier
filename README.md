# A Neuromorphic Network for Generic Multivariate Data Classification
This project implements the neuromorphic classifier network as described in [1].

In its current version, it requires the "Spikey" neuromorphic hardware system [2], that is developed at Kirchhoff-Institute for Physics, Heidelberg University [3].

How to run an example:

    $ cd src
    $ python mnist_classifier_on_spikey.py --help
    Usage: mnist_classifier_on_spikey.py [options]
    
    Options:
      -h, --help            show this help message and exit
      -s WORKSTATION, --station=WORKSTATION
                            spikey workstation to use
      -n NUM_DATA_SAMPLES, --num_data_samples=NUM_DATA_SAMPLES
                            number of data samples from each class to use
    $ python mnist_classifier_on_spikey.py -n 200 -s station112 
    [...]

The above will only work if you have access to a 'Spikey' chip. If you'd rather like to run it using a simulator, and are willing to help in the porting process, please raise an issue with the repository. 

Porting shouldn't be that difficult if you have some PyNN experience. I'll be glad to assist. 

### References 
[1] Schmuker, M.; Pfeil, T.; Nawrot, M. P. A Neuromorphic Network for Generic Multivariate Data Classification. Proc. Natl. Acad. Sci. U. S. A. 2014, 111, 2081–2086. http://www.pnas.org/cgi/doi/10.1073/pnas.1303053111 .

[2] Pfeil, T.; Grübl, A.; Jeltsch, S.; Müller, E.; Müller, P.; Petrovici, M. A.; Schmuker, M.; Brüderle, D.; Schemmel, J.; Meier, K. Six Networks on a Universal Neuromorphic Computing Substrate. Front. Neurosci. 2013, 7, 11. http://dx.doi.org/10.3389/fnins.2013.00011 .

[3] http://www.kip.uni-heidelberg.de/cms/vision/projects/facets/neuromorphic_hardware/single_chip_system/the_spikey_chip/
