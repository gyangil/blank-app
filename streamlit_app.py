import streamlit as st
import pandas as pd
import numpy as np

# Load similarity matrix
def load_similarity_matrix_local():
    return pd.read_csv("s_matrix_top30.csv", index_col=0)

# Normalize similarity matrix index
def normalize_similarity_matrix(similarity_matrix):
    similarity_matrix.index = similarity_matrix.index.str.lstrip('m')
    similarity_matrix.columns = similarity_matrix.columns.str.lstrip('m')
    return similarity_matrix

# Load movies.dat file
def load_movies():
    movies = pd.read_csv(
        "movies.dat", 
        sep="::", 
        engine="python", 
        names=["id", "title", "genres"], 
        index_col="id", 
        encoding="latin-1"  # Specify the encoding to handle special characters
    )
    return movies

# Recommendation function
def myIBCF(new_user_ratings, similarity_matrix, top_n=10):
    user_ratings = pd.Series(new_user_ratings)
    predictions = {}

    for movie_id in similarity_matrix.index:
        if pd.notna(user_ratings.get(int(movie_id), np.nan)):  # Ensure numeric comparison
            continue

        # Retrieve similarity values and user-rated movies
        sim_scores = similarity_matrix.loc[movie_id]
        rated_movies = [str(mid) for mid in user_ratings[user_ratings.notna()].index]

        # Filter relevant similarities and ratings
        relevant_sims = sim_scores[rated_movies].dropna()
        relevant_ratings = user_ratings[relevant_sims.index.astype(int)]

        if not relevant_sims.empty:
            weighted_sum = np.dot(relevant_sims, relevant_ratings)
            sim_sum = relevant_sims.sum()
            predictions[movie_id] = weighted_sum / sim_sum if sim_sum != 0 else np.nan

    # Sort predictions and return top N
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return [int(movie) for movie, _ in sorted_predictions[:top_n]]

# Main application
def main():
    st.title("Movie Recommender")
    st.markdown("## Step 1: Rate as many movies as possible")

    # Load and normalize the similarity matrix
    st.write("Loading similarity matrix...")
    similarity_matrix = load_similarity_matrix_local()
    similarity_matrix = normalize_similarity_matrix(similarity_matrix)

    # Load movies data
    st.write("Loading movies...")
    movies_df = load_movies()

    # Pagination for movies
    page = st.number_input("Page", min_value=1, max_value=(len(movies_df) // 6) + 1, step=1, value=1)
    start_idx = (page - 1) * 6
    end_idx = start_idx + 6

    # Display 6 movies per page
    user_ratings = {}
    cols = st.columns(3)
    for idx, (movie_id, row) in enumerate(movies_df.iloc[start_idx:end_idx].iterrows()):
        with cols[idx % 3]:
            poster_url = f"https://liangfgithub.github.io/MovieImages/{movie_id}.jpg"
            st.image(poster_url, caption=row['title'], use_container_width=True)
            user_ratings[movie_id] = st.radio(
                f"Select a rating for {row['title']}",
                options=["Select a rating", "★", "★★", "★★★", "★★★★", "★★★★★"],
                index=0
            )

    # Convert ratings to numerical values
    for movie_id in user_ratings:
        rating = user_ratings[movie_id]
        user_ratings[movie_id] = {"★": 1, "★★": 2, "★★★": 3, "★★★★": 4, "★★★★★": 5}.get(rating, np.nan)

    # Process user input
    if st.button("Get Recommendations"):
        st.write("Generating recommendations...")

        # Generate recommendations
        recommendations = myIBCF(user_ratings, similarity_matrix)

        st.header("Top 10 Movie Recommendations")
        for i, movie_id in enumerate(recommendations, start=1):
            st.write(f"{i}. {movies_df.loc[int(movie_id), 'title']}")

if __name__ == "__main__":
    main()
