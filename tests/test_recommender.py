"""
test_recommender.py - tests for the AI part (recommender.py).

Each function that starts with "test_" is one test.
"assert" means: "this must be true, otherwise the test FAILS".
"""

import pytest

# ------------------------------------------------------------------
# 1. The data
# ------------------------------------------------------------------

def test_data_has_the_right_size(data):
    movies, ratings = data
    assert len(movies) == 9742
    assert len(ratings) == 100836


def test_data_has_the_right_columns(data):
    movies, ratings = data
    assert list(movies.columns) == ["movieId", "title", "genres"]
    assert list(ratings.columns) == ["userId", "movieId", "rating", "timestamp"]


def test_ratings_are_between_half_and_five_stars(data):
    _, ratings = data
    assert ratings["rating"].min() == 0.5
    assert ratings["rating"].max() == 5.0


# ------------------------------------------------------------------
# 2. Content-based model (cosine similarity)
# ------------------------------------------------------------------

def test_similar_movies_returns_10_movies(content):
    result = content.similar_movies("Toy Story (1995)")
    assert len(result) == 10
    assert list(result.columns) == ["title", "genres", "similarity"]


def test_similar_movies_does_not_recommend_the_movie_itself(content):
    result = content.similar_movies("Toy Story (1995)")
    assert "Toy Story (1995)" not in result["title"].values


def test_similarity_scores_are_between_0_and_1(content):
    result = content.similar_movies("Matrix, The (1999)")
    # a tiny margin, because computers round decimals (1.0000000000000002)
    assert result["similarity"].between(0, 1 + 1e-9).all()


def test_similar_movies_are_sorted_best_first(content):
    scores = content.similar_movies("Jumanji (1995)")["similarity"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_toy_story_2_is_100_percent_similar_to_toy_story(content):
    # Same genres -> cosine similarity = 1.0
    result = content.similar_movies("Toy Story (1995)")
    toy_story_2 = result[result["title"] == "Toy Story 2 (1999)"]
    assert len(toy_story_2) == 1
    assert toy_story_2["similarity"].iloc[0] == pytest.approx(1.0)


def test_similar_movies_share_genres(content):
    # Every movie similar to an animation film should also be an animation film
    result = content.similar_movies("Toy Story (1995)")
    assert result["genres"].str.contains("Animation").all()


def test_n_controls_how_many_movies_come_back(content):
    assert len(content.similar_movies("Toy Story (1995)", n=3)) == 3


def test_unknown_movie_raises_an_error(content):
    # recommender.py doesn't check the title itself; app.py checks it before calling.
    with pytest.raises(IndexError):
        content.similar_movies("This Movie Does Not Exist (2099)")


# ------------------------------------------------------------------
# 3. SVD model (collaborative filtering)
# ------------------------------------------------------------------

def test_svd_hides_20_percent_of_ratings_for_testing(svd, data):
    _, ratings = data
    assert len(svd.testset) == pytest.approx(0.2 * len(ratings), abs=5)


def test_svd_rmse_is_good(svd):
    # 0.861 with our settings. Always guessing the average gives 1.045.
    assert svd.rmse() < 0.9


def test_svd_beats_always_guessing_the_average(svd):
    truth = [rating for (_, _, rating) in svd.testset]
    average = svd.trainset.global_mean
    baseline = (sum((r - average) ** 2 for r in truth) / len(truth)) ** 0.5
    assert svd.rmse() < baseline


def test_svd_gives_the_same_result_every_time(svd, data):
    # random_state=42 -> training twice gives exactly the same model
    from recommender import SVDModel
    movies, ratings = data
    assert SVDModel(movies, ratings).rmse() == pytest.approx(svd.rmse())


def test_recommend_returns_10_movies(svd):
    result = svd.recommend(1)
    assert len(result) == 10
    assert list(result.columns) == ["title", "genres", "predicted rating"]


def test_recommend_only_unseen_movies(svd, data):
    movies, ratings = data
    seen_ids = set(ratings.loc[ratings["userId"] == 1, "movieId"])
    seen_titles = set(movies.loc[movies["movieId"].isin(seen_ids), "title"])
    recommended = set(svd.recommend(1)["title"])
    assert recommended.isdisjoint(seen_titles)


def test_predicted_ratings_are_valid_stars(svd):
    result = svd.recommend(5)
    assert result["predicted rating"].between(0.5, 5.0).all()


def test_recommendations_are_sorted_best_first(svd):
    scores = svd.recommend(42)["predicted rating"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_different_users_get_different_recommendations(svd):
    assert set(svd.recommend(1)["title"]) != set(svd.recommend(2)["title"])
