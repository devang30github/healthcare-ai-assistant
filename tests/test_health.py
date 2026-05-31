"""
Tests for GET /health endpoint.
"""
import pytest


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_response_shape(client):
    r    = client.get("/health")
    data = r.json()
    assert "status"              in data
    assert "vector_store_loaded" in data
    assert "environment"         in data


def test_health_status_ok(client):
    r    = client.get("/health")
    data = r.json()
    assert data["status"] == "ok"


def test_health_environment_set(client):
    r    = client.get("/health")
    data = r.json()
    assert data["environment"] in ("development", "production", "test")


def test_health_vector_store_is_bool(client):
    r    = client.get("/health")
    data = r.json()
    assert isinstance(data["vector_store_loaded"], bool)
