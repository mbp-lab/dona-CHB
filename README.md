# What is this repository about?

This code reproduces the analysis in the paper "Know Thyself Through Data: Improving WhatsApp Interaction Awareness with Data-Driven Visualizations" by Hakobyan and Drimalla accepted for publication at Computers in Human Behavior. 

## Paper abstract
Digital behaviors are often quantified through screen time or frequency metrics, providing insights
into activity patterns and their alignment with subjective perceptions. However, these metrics alone
fall short of capturing texting, a behavior involving complex interpersonal dynamics. Research on
messaging self-perception is scarce – a critical gap since biased perceptions can strain relationships
and affect well-being. In one of the first empirical studies on this topic, we conducted an online study
with 68 participants, who donated meta-data (message lengths and times) from over 300 WhatsApp
chats. Participants answered questions about their messaging behavior, such as whether they write
more or respond faster than their contacts. They then received personalized visualizations based on
donated data and reassessed their perceptions, allowing us to examine whether data-driven insights
improved self-report accuracy. The results show a significant improvement in the alignment between
self-reports and objective metrics for absolute response speed and participants’ contribution within
a chat. In contrast, assessments of peak activity times and response speed relative to contacts
were already fairly accurate and did not change. Finally, although self-report scores changed for
two subjective estimates – the even distribution of messages across chats and texting throughout
the day – their alignment with objective metrics did not improve significantly. To account for the
potential emotional impact of challenging participants’ existing self-perceptions, we assessed their
mood and did not find significant changes in positive or negative affect. To conclude, personalized
visualizations can enhance the alignment between self-perceptions and objective behavior, potentially
guiding individuals toward more intentional texting behavior.

# Getting started with the code
The code is written in Python and structured into several Jupyter notebooks. Below you can find instructions on how to obtain the data, set up the environment and run the analyses. 
## Setup with conda

1. Start by cloning this repository somewhere on your machine.
2. The data of the study is available on Zenodo at [this URL](10.5281/zenodo.15504318). Please download all data and place it in the `data` folder.
3. Next, install conda. Instructions specific to your operating system can be found at the [conda website](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
4. After conda is set up, open a terminal/shell, navigate to the location where you cloned this repository to and create the Python environment with the command `conda env create -f dona-chb.yml -n dona-chb`.
5. With the environment created, activate it using `conda activate dona-chb`.
6. Lastly, use the command `jupyter lab` to open the IDE that runs the Jupyter notebooks with our analysis code.


## Reproducing the analysis
