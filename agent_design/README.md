# Agent Design

This directory contains the agent, and all scripts required to make it work.

_To run the agent, ensure all modules in the requirements file are installed._

This does not include any design experimentation, but a `production-ready` agent.

There are a number of files copied from the data analysis directory - the old versions are kept for completeness.

The files in this directory are as follows:

- .DB : Directory containing database and data for recommender to work
- Modules : Directory of all modules and scripts used by agent
  - scraping : Code which manages scraping
  - whiskynlp : Whisky specific KE and Lemmatizing scripts
  - types.py : script of Types which is imported to be used by the type hinter. Has no impact on te actual agent, just used to help program.
- Demo.ipynb : A jupyter notebook including a demo of all uses of the agent
- RecommenderAgent.py : The script containing the main recommender agent
- scotch.csv : The initial data set
- scotch-backup.csv : A backup of the initial dataset.
- survey_recommendations.ipynb : The script used to produce the recommendations for the survey. Note that this has been run since so the baseline recommendations are different to those used in survey. Should have set random state but had already started the survey at this point.
