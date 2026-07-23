import streamlit as st
import pandas as pd
import pickle
import difflib
import os
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Styling
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        height: 3em;
    }
    .movie-card {
        padding: 16px;
        border-radius: 10px;
        background-color: #1E222D;
        margin-bottom: 12px;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .movie-card h4 {
        color: #FFFFFF;
        margin-bottom: 6px;
    }
    .movie-card p {
        color: #B0B3B8;
        margin: 2px 0;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load Data & Artifacts with Caching
@st.cache_resource
def load_data():
    base_path = "artifacts"
    movies_path = os.path.join(base_path, "movies.pkl")
    similarity_path = os.path.join(base_path, "similarity.pkl")
    
    if not os.path.exists(movies_path):
        movies_path = "movies.pkl"
        similarity_path = "similarity.pkl"
        
    with open(movies_path, 'rb') as f:
        movies_df = pickle.load(f)
    with open(similarity_path, 'rb') as f:
        tfidf_matrix = pickle.load(f)
        
    return movies_df, tfidf_matrix

try:
    movies, tfidf_matrix = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    data_loaded = False

# 4. Header Section
st.title("🎬 End-to-End Movie Recommendation System")
st.markdown("Discover personalized movie recommendations using Content-Based TF-IDF Filtering.")

if data_loaded:
    movie_list = movies['title'].values

    # Sidebar controls
    st.sidebar.header("🔍 Search & Settings")
    selected_movie = st.sidebar.selectbox(
        "Select or type a movie title:",
        movie_list
    )

    top_n = st.sidebar.slider("Number of recommendations:", min_value=5, max_value=20, value=10)

    # Main Action Button
    if st.button("Get Recommendations"):
        with st.spinner("Finding similar movies..."):
            movie_title_clean = str(selected_movie).strip().lower()
            title_to_index = {str(title).lower(): idx for idx, title in enumerate(movies['title'])}
            all_titles = list(title_to_index.keys())

            if movie_title_clean in title_to_index:
                idx = title_to_index[movie_title_clean]
            else:
                close_matches = difflib.get_close_matches(movie_title_clean, all_titles, n=1, cutoff=0.3)
                if close_matches:
                    idx = title_to_index[close_matches[0]]
                else:
                    idx = None

            if idx is not None:
                # Memory-efficient on-the-fly similarity score calculation
                target_vector = tfidf_matrix[idx]
                sim_scores = cosine_similarity(target_vector, tfidf_matrix).flatten()
                
                # Get top indices excluding the query movie
                top_indices = [i for i in sim_scores.argsort()[::-1] if i != idx][:top_n]

                st.subheader(f"Top {top_n} Recommendations for '{movies.iloc[idx]['title']}':")
                st.write("---")
                
                # Render results in two grid columns
                cols = st.columns(2)
                for rank, m_idx in enumerate(top_indices, 1):
                    col = cols[(rank - 1) % 2]
                    rec_title = movies.iloc[m_idx]['title']
                    rec_genres = str(movies.iloc[m_idx]['genres']).replace('|', ', ')
                    score = sim_scores[m_idx]
                    
                    with col:
                        st.markdown(f"""
                        <div class="movie-card">
                            <h4>{rank}. {rec_title}</h4>
                            <p><b>Genres:</b> {rec_genres}</p>
                            <p><b>Similarity Match:</b> {score*100:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("Movie title not found.")

st.sidebar.markdown("---")
st.sidebar.info("Built with Python, Scikit-Learn & Streamlit.")
