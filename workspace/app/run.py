import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages_categorized_table', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Gerrix code
    # Data for the 1rst new visual related to the total number of medical messages
    # Here 3 categories are taken related to medical and hospital topics
    hospital_counts = df[df['hospitals'] == 1].shape[0]
    medical_help_counts = df[df['medical_help'] == 1].shape[0]
    medical_product_counts = df[df['medical_products'] == 1].shape[0]
    medical_category_counts = [hospital_counts,medical_help_counts,medical_product_counts]
    medical_category_names = ['hospital','medical help','medical products'] 
    
    # Data for the 2d new visual related to the total number of missing people
    # Here 3 categories are taken related to missing people or people that appear in a place without relatives
    search_and_rescue_counts = df[df['search_and_rescue'] == 1].shape[0]
    child_alone_counts = df[df['child_alone'] == 1].shape[0]
    missing_people_counts = df[df['missing_people'] == 1].shape[0]
    missing_category_counts = [search_and_rescue_counts,child_alone_counts,missing_people_counts]
    missing_category_names = ['search and rescue','child alone','missing people']
    
    # End Gerrix code
                                                      
    # There are 3 Graphs shown there
    # 1st Graph: Distribution of Message per Genres
    # 2nd Graph: Distribution of Messages related to Medical Categories
    # 3rd Graph: Distribution of Messages related to missing people or people that appear in one place without relatives
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        {
            'data': [
                Bar(
                    x=medical_category_names,
                    y=medical_category_counts
                )
                ],
            'layout': {
                'title': 'Message Counts Related to Medical Categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Medical Category"}
                }
        },
        {
            'data': [
                Bar(
                    x=missing_category_names,
                    y=missing_category_counts
                )
                ],
            'layout': {
                'title': 'Message Counts Related to Missing People',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Missing Category"}
                }
        }
        ]

    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()