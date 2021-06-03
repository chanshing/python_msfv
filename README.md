# README #

Python code for two phase, immiscible, incompressible flow through porous media. Code based on Aarnes, Gimse, and Lie, 'An Introduction to the Numerics of Flow in Porous Media using Matlab'.

**run_msfv_src_conv_test.py**
For fixed source and sink, we solve for varying coarse grid sizes and compare results to the fine scale solution. *Correction functions are implemented*.

**run_msfv_dir_conv_test.py**
For fixed pressures 1 and -1, we solve for varying coarse grid sizes and compare results to the fine scale solution.

**tests/run_vms_test.py**
Run variational multiscale method for Finite Volume.

**tests/run_msfv_bases_test.py**
Run and compare MSFV with different ways of computing the basis functions.

**runq5.py**
This will run the explicit solver (upstream_mod.py), with uniform permeability and porosity on the quarter-five spot problem.

**runq10wplot.py**
This will run the implicit solver (newtRaph_mod.py), loading the permeability and porosity from a file called 'spe10.mat'.