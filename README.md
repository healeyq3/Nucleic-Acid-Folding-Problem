# Overview

This repository contains a project I completed for Georgia Tech's ISYE 3133 - Engineering Optimization. The assignment given to us was the Nucleic Acid folding problem, an important and classic problem in computational biology that is often solved with variants of dynamic programming. For this project, however, we had to use mixed integer linear programs to solve a simple folding problem, as well as folding problems which were extended to capture more realistic biological features.

Please note that I have included *my* formulations and code - not the instructors. With that being said, with the benefit of hindsight, I will be the first to admit that my work, particularly my MILP formulation, is rather verbose and unnecessarily complicated. In some cases, parts of my formulation *are wrong.* Consequently, the Python implementation of the models contains both my formulations, as well as formulations provided by the TAs. However, it is worth stating that the TAs also were not providing adequate MILPs, and in some cases the Gurobi models are too large to run on one's personal machine.

# Files

Assignment.pdf outlines the project as a whole and was provided by the instructors.

ReportFormulation.pdf contains my various MILP models.

The Code folder contains both a single Python file, as well as a Jupyter Notebook. Both of these sources contain the same code, but I primarily worked in the Jupyter Notebook.

ImplementationAnalysis.pdf 