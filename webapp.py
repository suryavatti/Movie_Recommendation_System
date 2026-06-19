import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.title("🎬 Movie Recommendation System")
# Load dataset
df = pd.read_csv("movies.csv")

df = df[['title', 'genres', 'overview']]
df.dropna(inplace=True)

df['tags'] = df['genres'] + df['overview']

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie_name):

    movie_index = df[df['title'].str.lower() == movie_name.lower()].index

    if len(movie_index) == 0:
        return ["Movie not found"]

    movie_index = movie_index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    result = []

    for movie in movies_list:
        result.append(df.iloc[movie[0]].title)

    return result



movie = st.text_input("Enter Movie Name")

if st.button("Recommend"):
    movies = recommend(movie)

    st.subheader("Recommended Movies")

    for m in movies:
        st.write("✅", m)