# What is this repository about?

This code reproduces the analysis in the paper "Know Thyself Through Data: Improving WhatsApp Interaction Awareness with Data-Driven Visualizations" by Hakobyan and Drimalla accepted for publication at Computers in Human Behavior. 


# Getting started with the code
The code is written in Python and structured into several Jupyter notebooks. This project was tested with Python 3.8, Conda 24.1 or higher. 
Below you can find instructions on how to set up the environment and run the analyses. 
## Setup with conda
1. Start by cloning this repository somewhere on your machine.
3. Next, install conda. Instructions specific to your operating system can be found at the [conda website](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
4. After conda is set up, open a terminal/shell, navigate to the location where you cloned this repository to and create the Python environment with the command `conda env create -f dona-chb.yml -n dona-chb`.
5. With the environment created, activate it using `conda activate dona-chb`.
6. Lastly, use the command `jupyter lab` to open the IDE that runs the Jupyter notebooks with our analysis code.

Alternatively, you can use the `requirements.txt` file for setup. 

## Reproducing the analysis
To reproduce the figures and numbers of the paper you need to:
1) Get the data - available on Zenodo at [this URL](10.5281/zenodo.15504318). Please download all data and place it in the `data` folder.
2) Run the pre-processing steps under code/1-preprocessing. This will result in data files (data/preprocessed) which will be used for further analysis.
3) Run the analysis scripts under code/2-analysis. You can run the anayses in any order if you have already completed the pre-processing steps from step 1.
4) Check out code/modules for statistical tests and plotting routines.

All figures are will be saved under reports. 

## Contact
If you have questions or encounter any problems, feel free to contact Dr. Olya Hakobyan at olya.hakobyan@uni-bielefeld.de or Prof. Dr. Hanna Drimalla at drimalla@techfak.uni-bielefeld.de


