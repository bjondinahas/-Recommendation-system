// script.js - makes the page work.
// When you click a button, it asks the backend (app.py) for data with fetch(),
// then turns the answer into movie cards.

// Find the elements on the page that we need
const movieInput = document.getElementById("movie-input");
const userInput = document.getElementById("user-input");
const resultsTitle = document.getElementById("results-title");
const results = document.getElementById("results");

// 1. When the page opens: load all movie titles for the search suggestions
fetch("/api/movies")
  .then(response => response.json())
  .then(titles => {
    const list = document.getElementById("titles");
    list.innerHTML = titles.map(title => `<option value="${escapeHtml(title)}">`).join("");
  });

// 2. "Find similar" button -> content-based model
document.getElementById("movie-button").addEventListener("click", () => {
  const title = movieInput.value.trim();
  // encodeURIComponent makes the title safe to put in a URL (spaces, "&", etc.)
  getMovies("/api/similar?movie=" + encodeURIComponent(title),
            `Movies similar to ${title}`, "similarity");
});

// 3. "Recommend" button -> SVD model
document.getElementById("user-button").addEventListener("click", () => {
  const user = userInput.value.trim();
  getMovies("/api/recommend?user=" + encodeURIComponent(user),
            `Recommendations for user ${user}`, "predicted rating");
});

// Pressing Enter in a box does the same as clicking its button
movieInput.addEventListener("keydown", e => { if (e.key === "Enter") document.getElementById("movie-button").click(); });
userInput.addEventListener("keydown", e => { if (e.key === "Enter") document.getElementById("user-button").click(); });

// Ask the backend for movies and show them as cards.
// scoreName is the field to show under each movie: "similarity" or "predicted rating".
async function getMovies(url, title, scoreName) {
  resultsTitle.textContent = "";
  results.innerHTML = '<p class="message">Loading...</p>';

  const response = await fetch(url);   // send the request to app.py
  const data = await response.json();  // read the JSON answer

  if (!response.ok) {                  // the backend sent an error message
    results.innerHTML = `<p class="error">${escapeHtml(data.error)}</p>`;
    return;
  }

  resultsTitle.textContent = title;
  results.innerHTML = data.map(movie => makeCard(movie, scoreName)).join("");
}

// Build the HTML for one movie card
function makeCard(movie, scoreName) {
  const genres = movie.genres.split("|")
    .map(g => `<span class="genre">${escapeHtml(g)}</span>`).join("");
  const score = scoreName === "similarity"
    ? `Similarity: ${Math.round(movie.similarity * 100)}%`
    : `Predicted rating: ${movie["predicted rating"].toFixed(1)} ★`;

  return `
    <div class="card">
      <h3>${escapeHtml(movie.title)}</h3>
      <div>${genres}</div>
      <p class="score">${score}</p>
    </div>`;
}

// Movie titles can contain characters like < or & — this makes them safe to show as text
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
