# DL4Baseball: Transfer Learning for Baseball Event Detection and Player Tagging 
By: Jonathan Herman and Aryeh Zapinsky

### Goal: 
The goal of the project is to set up a system that will automatically collect and label a dataset of batters at the plate.  
The purpose of this is to facilitate data collection to make deep learning in sports more accessible. 

### Project Structure:
./models/: These are the trained classifiers.  There are 2 models: one to detect names, one to detect at-bats.
./notebooks/: This directory contains Jupyter notebooks documenting how the networks were built and trained. 
./devel/: This contains code that we wrote that didn't make it into the final cut.  Many of these functions were incorporated into capture.py 
./capture.py: The data collection and preprocessing portion of our pipeline. 
./report/: Here we present our findings.  Both in the form of slides and a conference paper.

### Division of Labor: 
- Image capturing 
- Image preprocessing: Aryeh
- Labelling name data: Jon
- Labelling at-bat data: Aryeh
- Building and training at-bat detector: Jon
- Building and training name detector: Aryeh
- Collecting and labeling second round of name data: Aryeh  # edit this
- Collecting and labeling second round of at-bat data: Jon

##### Putting it all together:
- Handling concurrency: Aryeh
- Hooking up classifiers: Jon
  
##### Next steps:
- Collecting baseball statistics: Jon
