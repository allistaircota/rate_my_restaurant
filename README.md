# Rate My Restaurant
## BrainStation Data Science Bootcamp Capstone Project
*Allistair Cota*

*September 2021 Cohort*

This is the repository for my capstone project for BrainStation's Data Science Bootcamp.

The focus of this project was to build a regression model that can predict restaurant review ratings.

This folder contains the following:

- This README file.
- 1 requirements.txt file that lists all the required packages for the project environment.
- 1 folder titled "notebooks" which contains the following:
  - NB1-Project_Intro_and_Data_Loading.ipynb
  - NB2-Data_Cleaning_EDA_Feature_Engineering.ipynb
  - NB3-Model_Setup_and_Evaluation.ipynb
  - Numerous pkl files, of fitted models and grid searches
- 1 folder titled "data" which contains the following cleaned data files:
    - ma_restaurant.csv
    - ma_review.csv
    - review_data_cleaned.csv

The original dataset files can be obtained directly from the [Yelp Open Dataset website](https://www.yelp.com/dataset).

Please open the Jupyter notebooks in the 'notebooks' folder, in the following sequence:
- NB1-Project_Intro_and_Data_Loading.ipynb
- NB2-Data_Cleaning_EDA_Feature_Engineering.ipynb
- NB3-Model_Setup_and_Evaluation.ipynb

Note while NB1-Project_Intro_and_Data_Loading.ipynb must be read, the code will not run as it loads data from the original JSON files from the Yelp website which are very large in size (~7GB).

The resulting filtered datasets that were written at the end of Project_Intro_and_Data_Loading.ipynb are included in the data folder, allowing the code in NB2-Data_Cleaning_EDA_Feature_Engineering.ipynb and NB3-Model_Setup_and_Evaluation.ipynb to be run.

For questions, please feel free to contact me at allistair.cota@gmail.com.




