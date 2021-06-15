# Data Analysis and Preprocessing

This directory contains the initial scripts used for the following purposes:

- Scraping masterofmalt.com

- Experimenting with feature selection

- Initial prototypical recommendation engine

Many parts of this code is copied in the agent design folder - this is so that the design process is hopefully more clear to a reader/marker. The rough scripts and functions in here are not expected to be `production-ready`; they are kept for completeness.

The following files are in this folder:

- scrape-scripts : scripts used for initial web scraping
- whiskynlp : whisky specific NLP scripts
- clustering.ipynb : Used to produce clusters for sanity checking of BoW models
- kwe-comparison.ipynb, kwe-comparison-2.ipynb, kwe-comparison.pdf : Used to compare kwe techniques
- results.json, txt-summary.txt : Results from kwe comparison
- scotch-no-dupes.csv : Initial dataset after removing duplicates
- scotch-preprop.ipynb : preprocessing, removing duplicates and bits of exploration
- scotch.csv : initial dataset with duplicates
- vectorised_all.csv : Saved models from initial exploration
