# Restaurant Rating Predictor
## BrainStation Data Science Bootcamp Capstone Project
*Allistair Cota*

*September 2021 Cohort*


This is the repository for my capstone project for BrainStation's Data Science Bootcamp.

The focus of this project was to build a regression model that can predict restaurant review ratings. The data was obtained from the [Yelp Open Dataset](https://www.yelp.com/dataset).

This folder contains the following:

- This README file.
- 1 requirements.txt file that lists all the required packages for the project environment.
- 1 folder titled "notebooks" which contains the following:
  - NB1-Project_Intro_and_Data_Loading.ipynb
  - NB2-Data_Cleaning_EDA_Feature_Engineering.ipynb
  - NB3-Model_Setup_and_Evaluation.ipynb
  - Numerous pkl files, of fitted models and grid searches
- 1 folder titled "data" which contains the following:
    - business.json
    - ma_restaurant.csv
    - ma_review.csv
    - review_data_cleaned.csv
    - review.json

Please execute the code in the Jupyter notebooks in the 'notebooks' folder, in the following sequence:
- NB1-Project_Intro_and_Data_Loading.ipynb
- NB2-Data_Cleaning_EDA_Feature_Engineering.ipynb
- NB3-Model_Setup_and_Evaluation.ipynb

Note while NB1-Project_Intro_and_Data_Loading must be read, running the code is optional and potentially time consuming since it only involves loading the data from the original Yelp dataset JSON files into CSV files, while filtering for restaruants in the state of Massachusetts and their associated reviews. The reviews JSON file is 6.94 GB.

For questions, please feel free to contact me at allistair.cota@gmail.com.




