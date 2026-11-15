"""
test_app.py - tests for the website backend (app.py).

`client` is a fake browser: it sends requests to the Flask app without
starting a real server. We check the status code and the JSON answer.
  200 = OK
  404 = not found (our app uses it for a wrong movie or user)
"""

# ------------------------------------------------------------------
# 1. The page and its files
# ------------------------------------------------------------------

def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Movie Recommender" in response.data


def test_css_and_javascript_load(client):
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/script.js").status_code == 200


# ------------------------------------------------------------------
# 2. /api/movies  (titles for the search box)
# ------------------------------------------------------------------

def test_movie_list_has_all_titles(client):
    titles = client.get("/api/movies").get_json()
    assert len(titles) == 9742
    assert "Toy Story (1995)" in titles


# ------------------------------------------------------------------
# 3. /api/similar  (content-based model)
# ------------------------------------------------------------------

def test_similar_returns_10_movies(client):
    response = client.get("/api/similar", query_string={"movie": "Toy Story (1995)"})
    assert response.status_code == 200
    movies = response.get_json()
    assert len(movies) == 10
    assert set(movies[0]) == {"title", "genres", "similarity"}


def test_similar_works_with_special_characters(client):
    # "&" has a special meaning in web addresses, so this checks the encoding works
    response = client.get("/api/similar", query_string={"movie": "Dumb & Dumber (Dumb and Dumber) (1994)"})
    assert response.status_code == 200
    assert len(response.get_json()) == 10


def test_similar_unknown_movie_gives_an_error(client):
    response = client.get("/api/similar", query_string={"movie": "Not A Movie (2099)"})
    assert response.status_code == 404
    assert "Movie not found" in response.get_json()["error"]


def test_similar_without_a_movie_gives_an_error(client):
    response = client.get("/api/similar")
    assert response.status_code == 404
    assert "error" in response.get_json()


# ------------------------------------------------------------------
# 4. /api/recommend  (SVD model)
# ------------------------------------------------------------------

def test_recommend_returns_10_movies(client):
    response = client.get("/api/recommend?user=1")
    assert response.status_code == 200
    movies = response.get_json()
    assert len(movies) == 10
    assert set(movies[0]) == {"title", "genres", "predicted rating"}


def test_recommend_first_and_last_user(client):
    assert client.get("/api/recommend?user=1").status_code == 200
    assert client.get("/api/recommend?user=610").status_code == 200


def test_recommend_bad_user_ids_give_an_error(client):
    for bad_user in ["0", "611", "9999", "abc", "-1", "1.5", ""]:
        response = client.get("/api/recommend", query_string={"user": bad_user})
        assert response.status_code == 404, f"user={bad_user!r} should be rejected"
        assert "User ID must be a number" in response.get_json()["error"]
