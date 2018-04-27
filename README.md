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

├── mlb_stats.py
│   ├── mss_test.py
│   ├── screenshot.py
│   ├── tester.py
│   └── threads.py
├── history
│   ├── fine_tune.csv
│   └── vgg_16_entire.csv
├── models
│   ├── at_bat_net.hdf5
│   ├── namenet.hdf5
│   ├── namenet_entire_best.hdf5
│   └── namenet_initial_best.hdf5
├── notebooks
│   ├── AtBatterNotebook.ipynb
│   ├── AtBatterNotebook.py
│   ├── PlayerNameNotebook.ipynb
│   └── PlayerNameNotebook.py
├── record.csv
├── report
│   ├── DL4Baseball.gslide

6 directories, 24 files



### Division of Labor:
- Image capturing: Aryeh
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
