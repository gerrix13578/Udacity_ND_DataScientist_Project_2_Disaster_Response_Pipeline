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
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM Messages_categorized_table",engine)
  
    X = df['message'] # The feature is the messages that we get
    Y = df.iloc[:,5:] # The target is the matrix with the categories
    category_names = Y.columns #column names of the matrix

    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
    ('vectorizer', CountVectorizer(tokenizer=tokenize)),  # Tokenization and vectorization
    ('tfidf',TfidfTransformer()),
    ('estimator', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10)))  # MultiOutputClassifier
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data
    y_pred = model.predict(X_test)

    report = []
    
    for i in range(y_pred.shape[1]):
         print(f"Classification Report for Output {Y_test.columns[i]}:")
         #print(f"Classification Report for Output {i + 1}:")
         print(classification_report(Y_test.iloc[:,i],y_pred[:,i]))
         report.append(classification_report(Y_test.iloc[:,i],y_pred[:,i]))


def save_model(model, model_filepath):
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