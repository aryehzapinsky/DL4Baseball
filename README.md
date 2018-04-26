# DL4Baseball: Transfer Learning for Baseball Event Detection and Player Tagging
By: Jonathan Herman and Aryeh Zapinsky

[[https://github.com/username/repository/blob/master/img/octocat.png|alt=octocat]]

### Goal:
The goal of the project is to set up a system that will automatically collect and label a dataset of batters at the plate.
The purpose of this is to facilitate data collection to make deep learning in sports more accessible.

### Project Structure:
./models/: These are the trained classifiers.  There are 2 models: one to detect names, one to detect at-bats.

./notebooks/: This directory contains Jupyter notebooks documenting how the networks were built and trained.

./devel/: This contains code that we wrote that didn't make it into the final cut.  Many of these functions were incorporated into capture.py

./capture.py: The data collection and preprocessing portion of our pipeline.

./report/: Here we present our findings.  Both in the form of slides and a conference paper.

├── mlb_stats.py <br/>
│   ├── mss_test.py <br/>
│   ├── screenshot.py <br/>
│   ├── tester.py <br/>
│   └── threads.py <br/>
├── history <br/>
│   ├── fine_tune.csv <br/>
│   └── vgg_16_entire.csv <br/>
├── models <br/>
│   ├── at_bat_net.hdf5 <br/>
│   ├── namenet.hdf5 <br/>
│   ├── namenet_entire_best.hdf5 <br/>
│   └── namenet_initial_best.hdf5 <br/>
├── notebooks <br/>
│   ├── AtBatterNotebook.ipynb <br/>
│   ├── AtBatterNotebook.py <br/>
│   ├── PlayerNameNotebook.ipynb <br/>
│   └── PlayerNameNotebook.py <br/>
├── record.csv <br/>
├── report <br/>
│   ├── DL4Baseball.gslide <br/>

6 directories, 24 files



### Division of Labor:
- Image capturing: Aryeh
>>>>>>> 229dcb09bb31b3a6cb8af99dc69ff90694c8bb10
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
