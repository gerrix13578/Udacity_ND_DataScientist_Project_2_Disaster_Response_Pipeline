# Import Libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load and merge disaster messages and categories datasets.

    INPUT
    messages_filepath (str): Path to the disaster_messages.csv file.
    categories_filepath (str): Path to the category_messages.csv fi

    OUTPUT
    pd.DataFrame: Merged DataFrame containing messages and their categories.
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    '''
    Clean data. Prepare the dataframe expanding columns that would be used afterwards to define the model

    INPUT
    pd.Dataframe: Dataframe containing the data to be cleaned
    OUTPUT
    pd.Dataframe: Dataframe containing the data cleaned
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand=True) 
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Replace category columns to the new ones    
    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    '''
    Save the pd.Dataframe to a SQL database

    INPUT
    pd.Dataframe: dataframe containing the data to be saved
    database_filename (str) - filename of the sql database to be saved 
    OUTPUT
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Messages_categorized_table', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        #Get the system arguments in order to execute the program. These arguments are taken with the execution of the Python script.
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        #Start the process of loading data to pd.Dataframe
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        #Start the process of cleaning data from pd.Dataframe
        print('Cleaning data...')
        df = clean_data(df)
        
        #Save data into a SQL database
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
