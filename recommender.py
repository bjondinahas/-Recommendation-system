"""
recommender.py - a simple movie recommender with two models:

  1. Content-based: "this movie has the same genres as that movie"
  2. Collaborative filtering (SVD): "people with similar taste liked this"

Run this file to see the RMSE of the SVD model and some example recommendations:
    python recommender.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

DATA = "data/ml-latest-small/"


 
# 1. Load the data

def load_data():
    """Read the two tables we need: movies and ratings."""
    movies = pd.read_csv(DATA + "movies.csv")    # movieId, title, genres
    ratings = pd.read_csv(DATA + "ratings.csv")  # userId, movieId, rating, timestamp
    return movies, ratings



# 2. Content-based model (genres + cosine similarity)

class ContentModel:
    """Finds movies with similar genres."""

    def __init__(self, movies, ratings):
        self.movies = movies.reset_index(drop=True)

        # Turn each movie's genres into a list of numbers (a "vector").
        # "Adventure|Animation|Children" -> one number per genre.
        # TF-IDF gives rare genres (e.g. "Film-Noir") more weight than
        # common ones (e.g. "Drama"), because rare genres say more about a movie.
        vectorizer = TfidfVectorizer(token_pattern=r"[^|]+")  # split on "|"
        self.vectors = vectorizer.fit_transform(self.movies["genres"])

        # How many ratings each movie has. Used to break ties:
        # many movies have exactly the same genres, so we show the well-known ones first.
        self.popularity = self.movies["movieId"].map(ratings["movieId"].value_counts()).fillna(0)

    def similar_movies(self, title, n=10):
        """Return the n movies whose genres are most similar to `title`."""
        row = self.movies.index[self.movies["title"] == title][0]

        # Cosine similarity = how much two vectors point in the same direction.
        # 1.0 = exactly the same genres, 0.0 = no genre in common.
        scores = cosine_similarity(self.vectors[row], self.vectors)[0]

        result = self.movies.assign(similarity=scores, popularity=self.popularity)
        result = result.drop(row)  # don't recommend the movie itself
        result = result.sort_values(["similarity", "popularity"], ascending=False)
        return result[["title", "genres", "similarity"]].head(n)

# 3. Collaborative filtering model (SVD)
class SVDModel:
    """Predicts how much a user will like a movie from everyone's ratings."""

    def __init__(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings

        # Put the ratings in the format the `surprise` library expects.
        data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]],
                                    Reader(rating_scale=(0.5, 5.0)))

        # Hide 20% of the ratings. The model never sees them during training;
        # we use them at the end to check how good its guesses are.
        # random_state=42 makes the split the same every time.
        self.trainset, self.testset = train_test_split(data, test_size=0.2, random_state=42)

        # SVD learns 100 hidden "taste" numbers for every user and every movie.
        # A rating is predicted by comparing the user's numbers with the movie's.
        #   n_epochs = how many times it goes through all the ratings
        #   lr_all   = learning rate: how big each correction step is
        #   reg_all  = regularization: stops it from memorizing the training data
        self.algo = SVD(n_factors=100, n_epochs=30, lr_all=0.01, reg_all=0.1, random_state=42)
        self.algo.fit(self.trainset)

    def rmse(self):
        """Average error on the hidden 20% of ratings (lower is better)."""
        predictions = self.algo.test(self.testset)
        return accuracy.rmse(predictions, verbose=False)

    def recommend(self, user_id, n=10):
        """Predict a rating for every movie the user hasn't seen; return the top n."""
        seen = set(self.ratings.loc[self.ratings["userId"] == user_id, "movieId"])
        unseen = self.movies[~self.movies["movieId"].isin(seen)].copy()
        unseen["predicted rating"] = [round(self.algo.predict(user_id, m).est, 2) for m in unseen["movieId"]]
        top = unseen.sort_values("predicted rating", ascending=False).head(n)
        return top[["title", "genres", "predicted rating"]]



# Try it out
if __name__ == "__main__":
    movies, ratings = load_data()
    print(f"{len(ratings):,} ratings, {ratings.userId.nunique()} users, {len(movies):,} movies\n")

    content = ContentModel(movies, ratings)
    print("Movies similar to Toy Story (1995):")
    print(content.similar_movies("Toy Story (1995)").to_string(index=False), "\n")

    svd = SVDModel(movies, ratings)
    print(f"SVD RMSE on the hidden 20% of ratings: {svd.rmse():.3f}\n")
    print("Top 10 recommendations for user 1:")
    print(svd.recommend(1).to_string(index=False))
