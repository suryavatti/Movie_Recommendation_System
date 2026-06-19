import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("movies.csv")

# Keep useful columns
df = df[['title', 'genres', 'overview']]

# Remove missing values
df.dropna(inplace=True)

# Create tags
df['tags'] = df['genres'] + df['overview']

# Convert text into vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie_name):

    movie_index = df[df['title'].str.lower() == movie_name.lower()].index

    if len(movie_index) == 0:
        print("Movie not found")
        return

    movie_index = movie_index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    print("\nRecommended Movies:\n")

    for movie in movies_list:
        print(df.iloc[movie[0]].title)

# User input
movie = input("Enter movie name: ")

recommend(movie)