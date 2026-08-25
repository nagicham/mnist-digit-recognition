import pytest
from app.inference import PARAMS_PATH, predict_digit

def test_predict_digit_returns_expected_keys(white_bg_black_digit_image):
    result = predict_digit(white_bg_black_digit_image)
    assert set(result.keys()) == {"predicted_digit", "confidence", "probabilities"}

def test_predicted_digit_is_valid_class(white_bg_black_digit_image):
    result = predict_digit(white_bg_black_digit_image)
    assert 0 <= result["predicted_digit"] <= 9

def test_confidence_is_probability(white_bg_black_digit_image):
    result = predict_digit(white_bg_black_digit_image)
    assert 0 <= result["confidence"] <= 1.0

def test_probabilities_sum_to_one(white_bg_black_digit_image):
    result = predict_digit(white_bg_black_digit_image)
    assert len(result["probabilities"]) == 10
    assert pytest.approx(sum(result["probabilities"]), abs=1e-3) == 1.0

def test_confidence_matches_max_probability(white_bg_black_digit_image):
    result = predict_digit(white_bg_black_digit_image)
    assert pytest.approx(result["confidence"], abs=1e-6) == max(result["probabilities"])
