from fastapi.testclient import TestClient
from app.main import app
c=TestClient(app)


def test_health():
    r=c.get("/healthz"); assert r.status_code==200 and r.json()["status"]=="ok"


def test_echo():
    r=c.get("/echo/hi"); assert r.json()=={"echo":"hi"}
