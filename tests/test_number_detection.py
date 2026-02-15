import pytest
import cv2
import numpy as np
from Number_Detect import NumberDetection


@pytest.fixture
def detector():
    return NumberDetection()


@pytest.fixture
def d6_crop():
    """A cropped die face showing the number 3."""
    img = cv2.imread(r"tests\test_images\d6_showing_3.jpg")
    return img


@pytest.fixture
def blank_crop():
    """A plain white image with no number."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 255


def test_debug(detector, d6_crop):
    """Visual debug — see what OCR is actually receiving."""
    print(f"\nCrop shape: {d6_crop.shape}")

    preprocessed = detector._preprocess(d6_crop)
    cv2.imshow("What OCR sees", preprocessed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    raw = detector.reader.readtext(preprocessed, detail=1)
    print(f"Raw OCR output: {raw}")


def test_reads_correct_number(detector, d6_crop):
    """Should read '3' from a D6 showing 3."""
    number, confidence = detector.read(d6_crop)
    assert number == "3"


def test_confidence_above_threshold(detector, d6_crop):
    """OCR confidence should be above 0.5 for a clear crop."""
    number, confidence = detector.read(d6_crop)
    assert confidence >= 0.5


def test_returns_none_on_blank_image(detector, blank_crop):
    """Should return None when there's no number to read."""
    number, confidence = detector.read(blank_crop)
    assert number is None


def test_returns_only_digits(detector, d6_crop):
    """Result should only contain digit characters."""
    number, confidence = detector.read(d6_crop)
    if number is not None:
        assert number.isdigit()