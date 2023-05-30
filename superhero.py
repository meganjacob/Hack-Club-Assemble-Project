import streamlit as st
import sklearn
import pandas as pd
import numpy as np
import math
from PIL import Image
import pickle
# import joblib
import random
from sklearn.base import BaseEstimator, TransformerMixin

# loading

DATA = 'heroes_information.csv'
POWERS = 'super_hero_powers.csv'

@st.cache
def load_data(file):
    data = pd.read_csv(file)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

# Filter dataframe for top closest to user inputted data
def closest(data, gender, eye_color, race, hair_color, height, publisher, skin_color, weight, top=3):
    filtered = data.loc[(data["gender"] == gender)
            & (data["eye color"] == eye_color)
            & (data["race"] == race)
            & (data["hair color"] == hair_color)
            & (data["publisher"] == publisher)
            & (data["skin color"] == skin_color)]
    if len(filtered) < top:
        filtered = data.loc[(data["gender"] == gender)
            & (data["eye color"] == eye_color)
            & (data["race"] == race)
            & (data["hair color"] == hair_color)
            & (data["publisher"] == publisher)]
        if len(filtered) < top:
            filtered = data.loc[(data["gender"] == gender)
            & (data["eye color"] == eye_color)
            & (data["race"] == race)
            & (data["hair color"] == hair_color)]
            if len(filtered) < top:
                filtered = data.loc[(data["gender"] == gender)
                & (data["eye color"] == eye_color)
                & (data["race"] == race)]
                if len(filtered) < top:
                    filtered = data.loc[(data["gender"] == gender)
                    & (data["eye color"] == eye_color)]
                    if len(filtered) < top:
                        filtered = data.loc[(data["gender"] == gender)]
    return filtered

class CustomRemover(BaseEstimator, TransformerMixin):

    def __init__(self, useless_attribs):
        self.useless_attribs = useless_attribs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        X_copy = X_copy.drop(self.useless_attribs, axis=1)

        return X_copy

# configuring streamlit

# full_pipeline = joblib.load('pipeline.joblib')

with open("pipeline.pkl", 'rb') as file2:
    full_pipeline = pickle.load(file2)

with open("model.pkl", 'rb') as file:
    clf = pickle.load(file)

st.title('If You Were a Superhero, Would You Be GOOD, NEUTRAL, or BAD?!?!')
 
# load data
data_load_state = st.text("Loading data...")
data = load_data(DATA)
powers = load_data(POWERS)
data_load_state.text('Loading data complete!!!')

st.write("Raw Data from Super Heroes Dataset from https://www.kaggle.com/datasets/claudiodavi/superhero-set")
st.write(data)

# user input
st.sidebar.subheader("Enter Your Information Here!")

name = st.sidebar.text_input("What is your superhero name?", placeholder ="Enter")

unnamed = st.sidebar.text_input("What is your sidekick's name?", placeholder ="Enter")

gender = st.sidebar.selectbox("What is your gender?", 
                                  ("Male", "Female"))

eye_color = st.sidebar.selectbox("What is your eye color?", 
                                  ("red", "blue", "green", "white", "Other"))

race = st.sidebar.selectbox("What is your race?", 
                                  ("Human", "Mutant", "Other"))

hair_color = st.sidebar.selectbox("What is your hair color?",
                                  ("Black", "No Hair", "Other"))

max_height = int(data["height"].max())
min_height = 0
height = st.sidebar.slider("What is your height in centimenters?", min_height, max_height, int((min_height+max_height)/2))

publisher = st.sidebar.selectbox("What is your favorite publisher?", 
                                ("Marvel Comics", "DC Comics", "Other"))

skin_color = st.sidebar.selectbox("What is your skin color?", 
                                  ("green", "blue", "white", "red", "Other"))

max_weight = int(data["weight"].max()) 
min_weight = 0
weight = st.sidebar.slider("What is your weight in pounds?", min_weight, max_weight, int((min_weight+max_weight)/2))


# get user data


# visitor counter
if 'number_submitted' not in st.session_state:
    st.session_state.number_submitted = 1

st.write("Number of Hack Clubbers Who Have Demoed Our Project: "+str(st.session_state.number_submitted))

# submit & predict
submit = st.button("Calculate my superhero affinity!")

if submit:
    user_input_prepared = pd.DataFrame(np.array([[unnamed, name, gender, eye_color, race, hair_color, height, publisher, skin_color, weight]]), columns =['Unnamed: 0', 'name', 'Gender', 'Eye color', 'Race', 'Hair color', 'Height', 'Publisher', 'Skin color', 'Weight'])
    st.write(user_input_prepared)
    user_input_prepared = full_pipeline.transform(user_input_prepared)

    pred = clf.predict(user_input_prepared)[0]
    
    if pred == 0:
         alignment = "BAD"
         image = Image.open('assets/bad.jpeg')
    elif pred == 1:
         alignment = "GOOD"
         image = Image.open('assets/good.jpeg')
    else:
        alignment = "NEUTRAL"
        image = Image.open('assets/neutral.jpeg')
        
    st.session_state.number_submitted+=1
    
    st.subheader("Here's the verdict...")

    st.write("Your alignment is...")
    st.subheader(alignment)
    st.image(image)

    # Retrieve closest 3 matches to what the user inputted and display
    search = closest(data, gender, eye_color, race, hair_color, height, publisher, skin_color, weight, top=3)

    st.write("And these are the top superheroes (or supervillains) who are most similar to you!")
    st.write(search)

    # To do: improve this section by only showing the cells that are True?
    st.write("These are some powers that would suit you based on these similar superheroes!")
    matches = search["name"]
    for match in matches:
        row_to_check = powers.loc[(powers["hero_names"] == match)]
        st.write(row_to_check)
        
        
        
