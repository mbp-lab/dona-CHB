# What is this repository about?

This code reproduces the analysis in the paper "Know Thyself Through Data: Improving WhatsApp Interaction Awareness with Data-Driven Visualizations" by Hakobyan and Drimalla accepted for publication at Computers in Human Behavior. 

The code is written in Python and structured into several Jupyter notebooks. Below you can find instructions on how to obtain the data, set up the environment and run the analyses. 

## Getting started
### Setup with conda

1. Start by cloning this repository somewhere on your machine.
2. The data of the study is available on Zenodo at [this URL](PLACEHOLDER). Please download all data and place it in the `data` folder.
3. Next, install conda. Instructions specific to your operating system can be found at the [conda website](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
4. After conda is set up, open a terminal/shell, navigate to the location where you cloned this repository to and create the Python environment with the command `conda env create -f dona-chb.yml -n dona-chb`.
5. With the environment created, activate it using `conda activate dona-chb`.
6. Lastly, use the command `jupyter lab` to open the IDE that runs the Jupyter notebooks with our analysis code.


### Reproducing the analysis
