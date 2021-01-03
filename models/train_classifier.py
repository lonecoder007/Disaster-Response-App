import sys
import numpy as np
import pandas as pd
import nltk
from sqlalchemy import create_engine
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,f1_score
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(database_filepath.replace('.db','')+"_table",engine)
    X = df['message']
    Y = df.iloc[:,4:]
    categories=Y.columns
    return X,Y,categories


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    words=word_tokenize(text.lower())
    stop=stopwords.words('english')
    lemi=WordNetLemmatizer()
    words=[lemi.lemmatize(word) for word in words]
    return words

def build_model():
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('mult_clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters ={'mult_clf__estimator__n_estimators':[10],
            'mult_clf__estimator__max_depth':[15],
            'mult_clf__estimator__criterion':['gini'],
            'mult_clf__estimator__min_samples_split':[2]}

    model = GridSearchCV(pipeline,param_grid=parameters,cv=5)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred=model.predict(X_test)
    # Print classification report on test data
    for i,col in enumerate(Y_test.columns):
        print(col)
        print(classification_report(Y_test[col],y_pred[:,i]))
    print((Y_test==y_pred).mean())

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