# LHW-IBW-dispersion-solver

This is a python code that calculates dispersion curves for lower hybrid waves and ion Bernstein waves in a magnetized plasma propagating perpendicularly to the background magnetic field.

All equations are derived in the paper "On perpendicularly propagating electromagnetic lower hybrid waves and ion Bernstein waves in a multi-component plasma".

The file DispRel.py contains the definitions of all the dispersion relations.

The file ComputeDispRel.py contains a method to evaluate the dispersion relations in DispRel.py

There are three jypyter notebooks that shows examples of how dispersion curves are calculated.
Example_EBW.ipynb:                         Evalyation of dispersion curves for EBWs
Example_LHW_IBW_single_species.ipynb:      Evalyation of dispersion curves for lower hybrid waves and ion Bernstein waves with a single ion species
Example_LHW_IBW_multiple_species.ipynb:    Evalyation of dispersion curves for lower hybrid waves and ion Bernstein waves with multiple ion species
