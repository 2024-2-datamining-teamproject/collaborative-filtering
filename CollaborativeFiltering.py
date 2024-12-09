import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

predicted_ratings_df = pd.read_csv('predicted_ratings.csv', index_col=0)
ratings = pd.read_csv('ratings.csv')

movie_ids = predicted_ratings_df.columns.astype(int)
user_ids = predicted_ratings_df.index.astype(int)

ratings_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
ratings_matrix = ratings_matrix.reindex(columns=movie_ids, fill_value=np.nan)
original_ratings_matrix = ratings_matrix.copy()

ratings_matrix = ratings_matrix.fillna(0)

def recommend_by_knn(user_id, top_n=10):
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5, n_jobs=-1)
    knn_model.fit(ratings_matrix.values)
    
    user_idx = ratings_matrix.index.get_loc(user_id)
    
    _, indices = knn_model.kneighbors([ratings_matrix.iloc[user_idx].values], n_neighbors=5)
    similar_users = ratings_matrix.index[indices.flatten()]
    #print("Similar User IDs: ", list(similar_users))
    
    recommendations = {}
    for similar_user in similar_users:
        similar_user_ratings = ratings_matrix.loc[similar_user]
        
        for movie_id, rating in similar_user_ratings.items():
            if pd.isna(original_ratings_matrix.loc[user_id, movie_id]) and not pd.isna(rating):
                if movie_id not in recommendations:
                    recommendations[movie_id] = rating
                else:
                    recommendations[movie_id] += rating
    
    recommended_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    
    return [int(movie_id) for movie_id, _ in recommended_movies[:top_n]]


if __name__ == "__main__":
    user_id = 2
    print("Movie Recommendations Using Collaborative Filtering:", recommend_by_knn(user_id=user_id))
