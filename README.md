# 🎬 End-to-End Movie Recommendation System

An end-to-end Machine Learning web application providing content-based movie recommendations using TF-IDF Vectorization and Cosine Similarity.

## Features
- Content-Based Recommendation: Matches genres across 60,000+ movies using high-dimensional space computations.
- Memory-Optimized Engine: On-the-fly similarity vector matrix processing to run efficiently on standard servers.
- Fuzzy Search Integration: Auto-resolves typos and spelling errors.
- Streamlit Web UI: Interactive dashboard with responsive layout.

## Dataset Structure
- movieId: Unique movie ID.
- title: Movie title and release year.
- genres: Pipe-separated genre tags (e.g., Action|Adventure|Sci-Fi).

## Local Running Instructions
1. Clone repository:
   git clone https://github.com/bhadkeswara16/movie-recommendation-system.git
   cd movie-recommendation-system

2. Install dependencies:
   pip install -r requirements.txt

3. Launch Streamlit Web App:
   streamlit run app.py

## Deploying to Render
1. Upload code to a GitHub Repository.
2. Go to Render.com and click New + -> Web Service.
3. Select your repository.
4. Set Build Command: pip install -r requirements.txt
5. Set Start Command: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
6. Click Deploy.
