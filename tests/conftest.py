import pytest
from PIL import Image, ImageDraw

@pytest.fixture
def white_bg_black_digit_image() -> Image.Image:
    """
    Black line on white background
    """
    img = Image.new("L", (200, 200), color=255)
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 40, 140, 160], outline=0, width=15)
    return img

@pytest.fixture
def blank_white_image() -> Image.Image:
    """
    White image with nothing drawn
    """
    return Image.new("L", (200, 200), color=255)
