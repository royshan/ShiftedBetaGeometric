ShiftedBetaGeometric
====================

The ShiftedBetaGeometric package provides an implementation of an extention of 
the Shifted-Beta-Geometric model by P. Fader & B. Hardie [1] to individuals in 
contractual settings with multiple predictor variables.

Installation
============

### Dependencies

* numpy
* pandas
* scipy

### Installation

ShiftedBetaGeometric is not currently available on PyPi. To install the package, 
you will need to clone it and run the setup.py file. Use the following commands to 
get a copy from Github and install all dependencies:

    git clone https://github.com/fmfn/ShiftedBetaGeometric.git
    cd ShiftedBetaGeometric
    python setup.py install

About
=====

ShiftedBetaGeometric offers a survival analysis like approach to modeling the 
behaviour of individuals in contractual settings. Given a set of features, ages
and status (alive or dead), this package builds on top of the model developed 
by P. Fader & B. Hardie [1], to infer hazard curves, retention curves and 
discounted-life-time-value.

Head over to the examples folder to learn how to use this package.


References
==========

1. "HOW TO PROJECT CUSTOMER RETENTION", P. Fader & B. Hardie
2. "Customer-Base Valudation in a Contractual Setting: The Perils of Ignoring 
Heterogeneity", P. Fader & B. Hardie

