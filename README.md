# Disaster Response Pipeline Project

Second project for the Udacity NanoDegree Data Scientist

Installations:
    Python 3.13.5
    Libraries: 
        pandas
        numpy
        sqlalchemy
        nltk
        pickle
        sklearn

### Project motivation:

    Predict the categories of a given message in order to classify it and afterwards alert the group responsible to attend this category.
    This project is prepared for Disaster Events when a lot of information, for example on social networks or direct calls to the emergency hotlines, are generated but it is very difficult to classify them manually. A tool to automatically classify this amounts of data helps the appropriate disaster relief agency to be aware which are the priorities and where is urgent to act.

### File Description:

    There are to files with data used in this project: 
        disaster_categories.csv
        disaster_messages.csv

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

    A web page is shown with 3 graphs that could be usefull to understand what categories the messages refer to.


### Licensing, Authors, Acknowledgements

    No licencsing required in this cas.