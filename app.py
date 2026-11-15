"""
app.py - the backend (Flask).

It trains the two models once when it starts, then answers requests
from the web page with JSON (plain data the JavaScript can read).

Run it with:
    python app.py
Then open http://127.0.0.1:5001 in your browser.
"""

from flask import Flask, jsonify, render_template, request

from recommender import ContentModel, SVDModel, load_data

app = Flask(__name__)

# Train both models once, when the server starts (takes a few seconds).
print("Training the models...")
movies, ratings = load_data()
content = ContentModel(movies, ratings)
svd = SVDModel(movies, ratings)
print("Ready!")


@app.route("/")
def home():
    """Send the web page (templates/index.html) to the browser."""
    return render_template("index.html")


@app.route("/api/movies")
def movie_titles():
    """All movie titles, used for the suggestions in the search box."""
    return jsonify(movies["title"].tolist())


@app.route("/api/similar")
def similar():
    """10 movies with similar genres. Example: /api/similar?movie=Toy Story (1995)"""
    title = request.args.get("movie", "")
    if title not in set(movies["title"]):
        return jsonify({"error": f"Movie not found: '{title}'"}), 404
    result = content.similar_movies(title)
    # .to_dict("records") turns the table into a list like [{"title": ..., "genres": ...}, ...]
    return jsonify(result.to_dict("records"))


@app.route("/api/recommend")
def recommend():
    """10 recommendations for a user. Example: /api/recommend?user=1"""
    user = request.args.get("user", "")
    if not user.isdigit() or int(user) not in set(ratings["userId"]):
        return jsonify({"error": "User ID must be a number from 1 to 610"}), 404
    result = svd.recommend(int(user))
    return jsonify(result.to_dict("records"))


if __name__ == "__main__":
    # Port 5001, because macOS already uses port 5000 for AirPlay
    app.run(port=5001, debug=False)
