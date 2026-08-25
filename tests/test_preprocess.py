import numpy as np
from app.preprocess import preprocess_image, CANVAS_SIZE

def test_output_shape(white_bg_black_digit_image):
    x = preprocess_image(white_bg_black_digit_image)
    assert x.shape == (1, 1, CANVAS_SIZE, CANVAS_SIZE)

def test_output_is_binarized(white_bg_black_digit_image):
    x = preprocess_image(white_bg_black_digit_image)
    unique_values = set(np.unique(x).tolist())
    assert unique_values.issubset({0.0, 1.0})

def test_digit_pixels_become_nonzero_after_invert(white_bg_black_digit_image):
    x_inverted = preprocess_image(white_bg_black_digit_image, invert=True)
    x_not_inverted = preprocess_image(white_bg_black_digit_image, invert=False)
    assert not np.array_equal(x_inverted, x_not_inverted)
    assert x_inverted.sum() > 0

def test_blank_image_does_not_crash(blank_white_image):
    x = preprocess_image(blank_white_image)
    assert x.shape == (1, 1, CANVAS_SIZE, CANVAS_SIZE)
