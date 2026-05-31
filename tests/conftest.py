import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="session")
def client():
    """Shared test client across all tests."""
    with TestClient(app) as c:
        yield c
