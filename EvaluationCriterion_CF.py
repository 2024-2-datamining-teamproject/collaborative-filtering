import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

predicted_ratings_df = pd.read_csv('predicted_ratings.csv', index_col=0)
ratings = pd.read_csv('ratings.csv')

predicted_ratings = predicted_ratings_df.values
movie_ids = predicted_ratings_df.columns.astype(int)
user_ids = predicted_ratings_df.index.astype(int)

ratings_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating', aggfunc='mean')
ratings_matrix = ratings_matrix.reindex(columns=movie_ids, fill_value=np.nan)
original_ratings_matrix = ratings_matrix.copy()

ratings_matrix = ratings_matrix.fillna(0)


def recommend_by_knn_with_rated(user_id, top_n=30):
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5, n_jobs=-1)
    knn_model.fit(ratings_matrix.values)
    
    user_idx = ratings_matrix.index.get_loc(user_id)
    
    _, indices = knn_model.kneighbors([ratings_matrix.iloc[user_idx].values], n_neighbors=5)
    similar_users = ratings_matrix.index[indices.flatten()]
    
    recommendations = {}
    for similar_user in similar_users:
        similar_user_ratings = ratings_matrix.loc[similar_user]
        
        for movie_id, rating in similar_user_ratings.items():
            if not pd.isna(rating):
                if movie_id not in recommendations:
                    recommendations[movie_id] = rating
                else:
                    recommendations[movie_id] += rating
                    
    recommended_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    
    return [int(movie_id) for movie_id, _ in recommended_movies[:top_n]]


def calculate_precision_recall(user_id, recommended_movies, k=30):
    # 실제 사용자가 좋아하는 영화 리스트 (평점 ≥ 4)
    actual_liked_movies = set(original_ratings_matrix.loc[user_id][original_ratings_matrix.loc[user_id] >= 4].index)
    
    top_k_recommendations = set(recommended_movies[:k])

    true_positives = len(top_k_recommendations & actual_liked_movies)
    precision = true_positives / len(top_k_recommendations) if len(top_k_recommendations) > 0 else 0

    return precision


if __name__ == "__main__":
    user_id = 1
    top_n = 10
    recommended_movies_knn = recommend_by_knn_with_rated(user_id=user_id, top_n=top_n)


    precision_knn = calculate_precision_recall(user_id, recommended_movies_knn, k=top_n)

    print("Evaluation Criterion of Collaborative Filtering")
    print(f"Base User ID: {user_id}, recommended movies: {recommended_movies_knn}")
    print(f"Precision@{top_n}: {precision_knn}")

