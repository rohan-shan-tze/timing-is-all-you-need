# A compact STDP-SNN

Our brain is super efficient, wouldnt it be cool if our machine could learn using similar mechanisms and achieve similar efficiencies? \
Spiking neural networks are neural networks which are designed to closely mimic the brain's neural network structure \
Spike-Timing-Dependent Plasticity is a learning rule that is evidenced to be used by our brain to build neural connections. 

This project is an implementation of a model inspired by Diel & Cook's 2015 paper on STDP.\
The goal is to show that the network size can be reduced from 6400 neurons while mostly maintaining accuracy on MNIST classification.\
Currently it has been shown that 30 neurons can achieve 75% accuracy.\
More experiments are being run to see if this can be improved via tuning hyperparameters

Additionally, other architectural choices have been explored, such as Difference of Gaussian preprocessing and a small Linear Regression readout layer, although these particular changes were needed to achieve 75% with 30 neurons.

This project was undertaken as part of my third year dissertation project at the University of Manchester's Computer Science Bsc(Hons) course.

Much thanks to my supervisor Luca Peres, my friends and family.