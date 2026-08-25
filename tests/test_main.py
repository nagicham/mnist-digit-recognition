import io
import pytest
from fastapi.testclient import TestClient
import app.main as main_module
from app.main import app

client = TestClient(app)

DUMMY_RESULT = {
    "predicted_digit": 7,
    "confidence": 0.987,
    "probabilities": [0.001, 0.002, 0.001, 0.001, 0.001, 0.003, 0.001, 0.987, 0.001, 0.001],
}

def _image_bytes(image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_returns_expected_schema(monkeypatch, white_bg_black_digit_image):
    monkeypatch.setattr(main_module, "predict_digit", lambda image: DUMMY_RESULT)
    response = client.post(
        "/predict",
        files={"file": ("digit.png", _image_bytes(white_bg_black_digit_image), "image/png")},
    )
    assert response.status_code == 200
    assert response.json() == DUMMY_RESULT

def test_predict_rejects_non_image_file():
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
    )
    assert response.status_code == 400

def test_predict_rejects_corrupted_image_data():
    response = client.post(
        "/predict",
        files={"file": ("fake.png", b"not actually png bytes", "image/png")},
    )
    assert response.status_code == 400
    
