import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import streamlit as st
from scipy.sparse import csr_matrix

# Load the rating matrix from a CSV file
# This matrix contains users as rows and movies as columns
try:
    rating_matrix = pd.read_csv("rating_matrix.csv", index_col=0)  # Assuming rows are users, columns are movies
    st.write("Rating matrix loaded successfully.")
except FileNotFoundError:
    st.error("rating_matrix.csv not found. Please ensure the file is in the correct directory.")
    st.stop()

# Step 1: Normalize the rating matrix
def normalize_matrix(matrix):
    """
    Normalize the matrix by centering each row (user).
    Ensures that user biases (e.g., some users always rate higher or lower) are accounted for.
    """
    return matrix.sub(matrix.mean(axis=1, skipna=True), axis=0)

# Step 2: Compute cosine similarity
def compute_similarity(matrix):
    """
    Compute the cosine similarity for movies with transformation.
    Calculates how similar each pair of movies is based on user ratings.
    The similarity values are transformed to the range [0, 1] for convenience.
    """
    movie_matrix = csr_matrix(matrix.fillna(0).T)  # Convert to sparse format
    similarity = cosine_similarity(movie_matrix)
    return pd.DataFrame(
        (1 + similarity) / 2,  # Transform similarity to [0, 1]
        index=matrix.columns,
        columns=matrix.columns
    )

# Step 3: Filter low-count similarities
def filter_low_counts(similarity_matrix, rating_matrix, min_shared=3):
    """
    Mask similarities with fewer than min_shared ratings.
    Ensures that the similarity calculation is based on enough data to be meaningful.
    """
    movie_matrix = rating_matrix.T.notna().astype(int)
    shared_counts = movie_matrix.T @ movie_matrix
    mask = shared_counts >= min_shared
    return similarity_matrix.where(mask, np.nan)

# Step 4: Keep top 30 similarities
def top_k_similarity(sim_matrix, k=30):
    """
    Keep only the top-k similarities for each movie.
    Reduces computational complexity and focuses on the most relevant similarities.
    """
    indices = np.argsort(-sim_matrix.values, axis=1)[:, :k]
    mask = np.zeros_like(sim_matrix.values, dtype=bool)
    rows = np.arange(sim_matrix.shape[0])[:, None]
    mask[rows, indices] = True
    return pd.DataFrame(
        np.where(mask, sim_matrix.values, np.nan),
        index=sim_matrix.index,
        columns=sim_matrix.columns
    )

# Step 5: Define the myIBCF function
def myIBCF(new_user_ratings, similarity_matrix, max_movies=100):
    """
    Generate movie recommendations for a new user.
    Limit the similarity matrix to only `max_movies` columns to save memory.
    Predictions are made based on the similarity scores and the ratings provided by the user.
    """
    try:
        limited_similarity = similarity_matrix.iloc[:, :max_movies]
        predictions = {}
        for movie in limited_similarity.index:
            sim_scores = limited_similarity[movie].dropna()
            rated_movies = new_user_ratings[new_user_ratings.notna()]
            overlapping_movies = sim_scores.index.intersection(rated_movies.index)
            if overlapping_movies.empty:
                predictions[movie] = np.nan
            else:
                weights = sim_scores[overlapping_movies]
                ratings = rated_movies[overlapping_movies]
                predictions[movie] = (weights @ ratings) / weights.sum()
        return pd.Series(predictions)
    except Exception as e:
        st.error(f"Error in myIBCF function: {str(e)}")
        return pd.Series()

# Preprocess data
try:
    start_time = datetime.now()
    normalized_ratings = normalize_matrix(rating_matrix)
    raw_similarity = compute_similarity(normalized_ratings)
    filtered_similarity = filter_low_counts(raw_similarity, rating_matrix)
    top_30_similarity = top_k_similarity(filtered_similarity)
    st.write(f"Data preprocessing completed in: {(datetime.now() - start_time).seconds / 60} minutes")
except Exception as e:
    st.error(f"Error during preprocessing: {str(e)}")
    st.stop()

# Streamlit App
st.title("Movie Recommendation System")
st.header("Rate Movies and Get Recommendations")

# User inputs for rating movies
movie_sample = rating_matrix.columns[:100]  # Limit to 100 movies for optimization
user_ratings = {}
st.write("Please rate the following movies (select ratings between 0 and 5):")

for movie in movie_sample:
    user_ratings[movie] = st.slider(f"Rate {movie}", min_value=0, max_value=5, value=0)

# Convert user ratings to a pandas Series
new_user = pd.Series(user_ratings)
new_user = new_user.replace(0, np.nan)  # Replace 0 ratings with NaN

# Generate recommendations
if st.button("Get Recommendations"):
    try:
        start_time = datetime.now()
        recommendations = myIBCF(new_user, top_30_similarity, max_movies=100)
        st.write(f"Recommendations generated in: {(datetime.now() - start_time).seconds / 60} minutes")
        st.write("Top 10 Recommendations:")
        st.table(recommendations.nlargest(10))
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")

# Note on memory optimization
st.write("""
**Note:** This application limits the recommendation process to the top 100 movies for memory and performance optimization. The similarity matrix is also constrained to these movies.
""")
