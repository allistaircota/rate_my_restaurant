# Rate My Restaurant
## BrainStation Data Science Bootcamp Capstone Project
*Allistair Cota*

*September 2021 Cohort*

This is the repository for my capstone project for BrainStation's Data Science Bootcamp.

The focus of this project was to build a regression model that can predict restaurant review ratings.

### Technical Summary
Refer to Capstone Final Report.pdf for a brief technical summary of the project.

### Jupyter Notebooks
Files related to Jupyter notebooks:
- notebook-requirements.txt file that lists all the required packages for the project environment.
- "notebooks" folder which contains the following:
  - NB1-Project_Intro_and_Data_Loading.ipynb
  - NB2-Data_Cleaning_EDA_Feature_Engineering.ipynb
  - NB3-Modelling.ipynb
  - Numerous pkl files, of fitted models and grid searches

The original dataset files can be obtained directly from the [Yelp Open Dataset website](https://www.yelp.com/dataset).

Please open the Jupyter notebooks in the 'notebooks' folder, in the following sequence:
- NB1-Project_Intro_and_Data_Loading.ipynb
- NB2-Data_Cleaning_EDA_Feature_Engineering.ipynb
- NB3-Modelling.ipynb

Note that the data files used in the notebooks are quite large (~10 GB) and are not uploaded to this repo. Retrieve the version 3 Yelp Open Dataset files directly from Yelp and run the notebooks in sequence to generate the required filtered CSV files.

### Web Application
Files related to web app:
- requirements.txt files that lists all the required packages for the Heroku app deployment
- nltk.txt file that lists the NLTK modules required by Heroku
- Procfile (required for Heroku app deployment)
- app.py script
- pipeline_fit.py script
- utils.py script
- "templates" folder which contains:
  - index.html
- linreg_pipeline.pkl
- X_train.csv (created in NB3-Modelling.ipynb)
- y_train.csv (created in NB3-Modelling.ipynb)

The web app is hosted on Heroku and can be visited here: http://rate-my-restaurant-app.herokuapp.com/.

To run the app on your local machine, type `flask run` into command line from the repo directory. 

For questions, please feel free to contact me at allistair.cota@gmail.com.




