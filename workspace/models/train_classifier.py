#Import libraries
import sys
import pandas as pd
import numpy as py
import nltk
import pickle
nltk.download('punkt')
nltk.download('wordnet')

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):

    '''
    Load data form a SQL Database created in the script process_data.py.

    INPUT
    database_filepath (str): Path to the SQL Database. For example: DisasterResponse.db

    OUTPUT
    pd.Dataframe X: Data used to predict the Y values. In this case the messages.
    pd.Dataframe Y: Data related directly to the X values and used to create the model
    list: list of the colum names of the matrix 
    '''

    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM Messages_categorized_table",engine)
  
    # Separate the data from the pd.Dataframe between the X and Y values to create afterwards the model
    X = df['message'] # The feature is the messages that we get
    Y = df.iloc[:,5:] # The target is the matrix with the categories
    category_names = Y.columns #column names of the matrix

    return X, Y, category_names


def tokenize(text):

    '''
    Tokenize a text with the important content in order to predict afterwards the category.

    INPUT
    text (str): content of a message

    OUTPUT
    clean_tokens (str): clean tokens 
    '''
    # Start the tokenize and lemmatize process
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # Clean the tokens using word_tokenize and lemmatizer
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    '''
    Pipeline creation in order to build the model. Here CountVectorizer, TfidTransformer, MultiOutputClassifier and RandomForestClassifier are used

    INPUT

    OUTPUT
    model: model definied by the function and improved using GridSearchCV
    '''
    # Pipepline creation    
    pipeline = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=tokenize)),  # Tokenization and vectorization
    ('tfidf',TfidfTransformer()),
    ('estimator', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10)))  # MultiOutputClassifier
    ])

    #Improve the model using
    parameters = {
    'vectorizer__max_df': [0.75, 1.0],
    'estimator__estimator__n_estimators': [10, 20],
    }

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters) 

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluation of the model and prepare a report

    INPUT
    Pipeline: Model based on a Pipepline
    X_test: Values used to predict the model based on a Pipeline
    Y_test: Values to test the model
    category_names: list of category names to be predicted

    OUTPUT
    '''
    # predict on test data
    y_pred = model.predict(X_test)

    # Create an empty list for a report
    report = []
    
    # Iterate througt the predicted values and compare it with the Y_test values to prepare the report
    for i in range(y_pred.shape[1]):
         print(f"Classification Report for Output {Y_test.columns[i]}:")
         #print(f"Classification Report for Output {i + 1}:")
         print(classification_report(Y_test.iloc[:,i],y_pred[:,i]))
         report.append(classification_report(Y_test.iloc[:,i],y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    Save the model to a pkl file

    INPUT
    model: Model based on a pipepline and improved
    model_filepath(str): filepath where the model will be stored as a pkl file

    OUTPUT
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
