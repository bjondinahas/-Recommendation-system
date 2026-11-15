"""
conftest.py - shared setup for all tests.

pytest reads this file automatically before running the tests.
"Fixtures" are things the tests need (data, trained models). Each one is
created ONCE (scope="session") and shared by all tests, so the models are
trained only once and the tests stay fast.
"""

import os
import sys

import pytest

# The project folder = the folder above tests/
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT)            # recommender.py looks for data/ in the current folder
sys.path.insert(0, PROJECT)  # so the tests can import recommender.py and app.py

from recommender import ContentModel, SVDModel, load_data  # noqa: E402


@pytest.fixture(scope="session")
def data():
    """The movies and ratings tables."""
    return load_data()


@pytest.fixture(scope="session")
def content(data):
    """A trained content-based model."""
    movies, ratings = data
    return ContentModel(movies, ratings)


@pytest.fixture(scope="session")
def svd(data):
    """A trained SVD model."""
    movies, ratings = data
    return SVDModel(movies, ratings)


@pytest.fixture(scope="session")
def client():
    """A fake browser that sends requests to the Flask app (no real server needed)."""
    import app
    return app.app.test_client()
