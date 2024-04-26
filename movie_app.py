# import pandas as pd
# import numpy as np
# import ast
# from nltk.stem import PorterStemmer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import pickle

# # Load datasets
# movies = pd.read_csv('tmdb_5000_movies.csv')
# credits = pd.read_csv('tmdb_5000_credits.csv')

# # Merge datasets on 'title' column
# movies = movies.merge(credits, on='title')

# # Keep only the relevant columns
# movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# # Remove rows with missing values
# movies.dropna(inplace=True)

# # Function to convert genres, keywords, cast, and crew from string to list
# # Function to convert genres, keywords, cast, and crew from string to list
# def convert_to_list(obj):
#     try:
#         # Convert the string representation of list to actual list
#         items = ast.literal_eval(obj)
#         return [item['name'] for item in items]
#     except:
#         # Return an empty list if there's an error during parsing
#         return []

# # Apply conversion function to genres and keywords
# movies['genres'] = movies['genres'].apply(convert_to_list)
# movies['keywords'] = movies['keywords'].apply(convert_to_list)

# # Function to fetch director's name from crew
# def fetch_director(obj):
#     try:
#         # Parse the string representation of crew list
#         crew_list = ast.literal_eval(obj)
#         for crew_member in crew_list:
#             if crew_member['job'] == 'Director':
#                 return [crew_member['name']]
#     except:
#         pass
#     return []

# # Apply fetch_director function to crew
# movies['crew'] = movies['crew'].apply(fetch_director)

# # Function to get the first three cast members' names
# def get_top_cast(obj):
#     try:
#         # Parse the string representation of cast list
#         cast_list = ast.literal_eval(obj)
#         return [cast_member['name'] for cast_member in cast_list[:3]]
#     except:
#         pass
#     return []

# # Apply get_top_cast function to cast
# movies['cast'] = movies['cast'].apply(get_top_cast)

# # Convert overview to list of words
# def convert_overview(overview):
#     if isinstance(overview, str):
#         return overview.split()
#     else:
#         return []

# movies['overview'] = movies['overview'].apply(convert_overview)

# # Combine overview, genres, keywords, cast, and crew into a single tags list
# def combine_tags(row):
#     tags = row['overview'] + row['genres'] + row['keywords'] + row['cast'] + row['crew']
#     return ' '.join(tags)

# movies['tags'] = movies.apply(combine_tags, axis=1)

# # Initialize PorterStemmer for stemming words
# ps = PorterStemmer()

# # Function to stem text
# def stem(text):
#     return ' '.join([ps.stem(word) for word in text.split()])

# # Apply stemming to tags
# movies['tags'] = movies['tags'].apply(stem)

# # Initialize CountVectorizer with maximum features and English stop words
# cv = CountVectorizer(max_features=10000, stop_words='english')

# # Generate count vectors for tags
# vectors = cv.fit_transform(movies['tags'])

# # Compute cosine similarity matrix
# similarity = cosine_similarity(vectors)

# # Save DataFrame and similarity matrix for later use
# pickle.dump(movies.to_dict(), open('movies_dict.pkl', 'wb'))
# pickle.dump(similarity, open('similarity.pkl', 'wb'))

# # Recommendation function
# def recommend(movie):
#     try:
#         movie_index = movies[movies['title'] == movie].index[0]
#         distances = similarity[movie_index]
#         movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
#         for i in movie_list:
#             print(movies.iloc[i[0]].title)
#     except IndexError:
#         print("Movie not found. Please ensure the title is spelled correctly.")






# all data print
import streamlit as st
import pickle
import pandas as pd
import requests
import gzip
import os

# Load data and similarity

# Load movie dictionary from compressed file
with gzip.open('movies_dict_compressed.pkl.gz', 'rb') as f:
    movie_dict = pickle.load(f)

# Load similarity data from compressed file
file_path = 'similarity_compressed.pkl.gz'
if not os.path.exists(file_path):
    st.error(f"File not found: {file_path}")
else:
    with gzip.open(file_path, 'rb') as f:
        similarity = pickle.load(f)

# Convert the movie dictionary to a pandas DataFrame
movies = pd.DataFrame(movie_dict)

# Function to fetch movie poster from the API
def fetch_poster(movie_id):
    # API call to fetch movie data
    response = requests.get(
        f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    )
    # Check if the response is successful
    if response.status_code != 200:
        st.error(f"Failed to fetch data for movie ID {movie_id}")
        return None
    data = response.json()
    # Return the URL of the movie poster if available
    if 'poster_path' in data:
        return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    else:
        return None

# Function to recommend movies based on the selected movie
def recommend(selected_movie):
    # Get the index of the selected movie
    try:
        movie_index = movies[movies['title'] == selected_movie].index[0]
    except IndexError:
        st.error(f"Movie not found: {selected_movie}")
        return [], []
    
    # Get the similarity scores for the selected movie
    similarity_scores = list(enumerate(similarity[movie_index]))

    # Sort the movies based on similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the recommended movies
    recommended_indices = [i[0] for i in similarity_scores[1:6]]

    # Get the names and posters of the recommended movies
    recommended_movie_names = movies.iloc[recommended_indices]['title'].values
    recommended_movie_posters = [fetch_poster(movies.iloc[i]['movie_id']) for i in recommended_indices]

    return recommended_movie_names, recommended_movie_posters

# Inject custom CSS code into the Streamlit app
st.markdown(
    """
    <style>
    .st-emotion-cache-gh2jqd {
        width: 100%;
        padding: 6rem 1rem 10rem;
        max-width: 95rem;
    }
    .st-emotion-cache-1663pn9 {
        width: 1114.4px;
        position: relative;
        display: flex;
        flex-direction: column;
        gap: 1rem;
        margin-top: -75px;
    }
    .st-emotion-cache-zt5igj {
        position: relative;
        left: calc(-3rem);
        width: calc(100% + 63rem);
        display: flex;
        align-items: center;
        overflow: visible;
    }
    .st-emotion-cache-1r4qj8v {
        position: absolute;
        background: black;
        color: black;
        inset: 0px;
        color-scheme: light;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True
)

# Columns layout
col1, col2 = st.columns([1, 6])

# Display header
col1.markdown("<h1 style='text-align: center; color: red;'>ND FILMS</h1>", unsafe_allow_html=True)

# Display a search box in the right column
selected_movie = st.selectbox("Type or select a movie from the dropdown", movies['title'].values)

# Flag to track whether the button has been clicked
show_recommendations = False

# Button to show recommendations
if st.button('Show Recommendation'):
    show_recommendations = True
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)

# Define the number of columns you want to display in a row for the movies
num_columns = 7

# Iterate over the movies dataset and display them side by side
if show_recommendations:
    # Display recommended movies
    st.write("Recommended Movies:")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])
else:
    # Display all movies
    for index, movie in movies.iterrows():
        # Determine which column to use for the current movie
        if index % num_columns == 0:
            columns = st.columns(num_columns)

        col = columns[index % num_columns]

        poster_url = fetch_poster(movie['movie_id'])
        col.image(poster_url, width=None, use_column_width=True)

        font_size = "10px"
        background_color = "red"
        padding = "5px"
        border_radius = "5px"
        margin_bottom="12px"
      
        # Display the movie title
        col.markdown(
            f"<div style='font-size: 10px; margin-bottom:12px; background-color: red; padding: 5px; border-radius: 5px; text-align: center;'>{movie['title']}</div>",
            unsafe_allow_html=True
        )






