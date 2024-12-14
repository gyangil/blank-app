import streamlit as st
import pandas as pd
import numpy as np

# Load similarity matrix
def load_similarity_matrix(url):
    return pd.read_csv(url, index_col=0)

# Recommendation function
def myIBCF(new_user_ratings, similarity_matrix, top_n=10):
    user_ratings = pd.Series(new_user_ratings)
    predictions = {}

    for movie_id in similarity_matrix.index:
        if pd.notna(user_ratings[movie_id]):
            continue

        # Retrieve similarity values and user-rated movies
        sim_scores = similarity_matrix[movie_id]
        rated_movies = user_ratings[user_ratings.notna()].index
        
        # Filter relevant similarities and ratings
        relevant_sims = sim_scores[rated_movies].dropna()
        relevant_ratings = user_ratings[relevant_sims.index]

        if not relevant_sims.empty:
            weighted_sum = np.dot(relevant_sims, relevant_ratings)
            sim_sum = relevant_sims.sum()
            predictions[movie_id] = weighted_sum / sim_sum if sim_sum != 0 else np.nan

    # Sort predictions and return top N
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return [movie for movie, _ in sorted_predictions[:top_n]]

# Main application
def main():
    st.title("Movie Recommender System")

    # Load the similarity matrix
    sim_matrix_url = "https://github.com/yh12chang/Cs_598_PSL_Project_4/raw/refs/heads/main/s_matrix_top30.csv"
    st.write("Loading similarity matrix...")
    similarity_matrix = load_similarity_matrix(sim_matrix_url)

    # Load sample movies
    movie_titles = similarity_matrix.index.tolist()[:100]  # Display only 100 movies

    # Display rating interface
    st.header("Rate Movies")
    user_ratings = {}
    for movie in movie_titles:
        user_ratings[movie] = st.slider(f"Rate {movie}", min_value=0, max_value=5, step=1, value=0)

    # Process user input
    if st.button("Get Recommendations"):
        st.write("Generating recommendations...")

        # Replace 0s with NaN for unrated movies
        user_ratings = {movie: (rating if rating > 0 else np.nan) for movie, rating in user_ratings.items()}

        # Generate recommendations
        recommendations = myIBCF(user_ratings, similarity_matrix)

        st.header("Top 10 Movie Recommendations")
        for i, movie in enumerate(recommendations, start=1):
            st.write(f"{i}. {movie}")

if __name__ == "__main__":
    main()
