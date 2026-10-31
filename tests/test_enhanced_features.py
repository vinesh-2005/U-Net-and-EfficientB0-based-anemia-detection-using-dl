"""Tests for modern medical UI, Grad-CAM explainability, and Evaluation Dashboard."""
import io
import json
import os
import cv2
import numpy as np
import pytest
from app import app, explainer, CLASSIFIER_THRESHOLD


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_page_title_and_medical_theme(client):
    """Verify application title, medical color palette, and navigation links."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")

    # Title verification
    assert "<title>Anemia Screening from Conjunctival Image</title>" in html
    assert "Anemia Screening from" in html
    assert "Conjunctival Image" in html

    # Medical theme color palette checks
    assert "#F0F7FA" in html
    assert "#0F766E" in html
    assert "#16324F" in html
    assert "#D8E6EC" in html
    assert "Evaluation Dashboard" in html


def test_dashboard_page(client):
    """Verify Evaluation Dashboard metrics, confusion matrix, and ROC curve."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    html = response.data.decode("utf-8")

    assert "Model Evaluation Dashboard" in html
    assert "Held-Out Test Set (N=29)" in html
    assert "Validation Set (N=31)" in html

    # Check verified metrics
    assert "79.31%" in html  # Test Accuracy
    assert "72.22%" in html  # Test Precision
    assert "92.86%" in html  # Test Recall
    assert "66.67%" in html  # Test Specificity
    assert "81.25%" in html  # Test F1
    assert "0.7762" in html  # Test ROC-AUC

    # Check confusion matrix counts
    assert "True Negative (TN)" in html
    assert "True Positive (TP)" in html

    # Check ROC SVG elements
    assert "rocSvg" in html
    assert "rocCurvePath" in html


def test_hb_reference_standalone(client):
    """Verify independent clinical Hb assessment still functions without an image."""
    response = client.post(
        "/",
        data={"sex": "female", "hb": "10.5", "age": "32"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    html = response.data.decode("utf-8")

    assert "Clinical Hb Reference Assessment" in html
    assert "Below Cutoff" in html
    assert "10.5" in html
    assert "Moderate Anemia" in html


def test_gradcam_explainer_execution():
    """Verify genuine Grad-CAM execution produces valid heatmap and overlay."""
    test_roi_path = "data/processed/roi_predicted_v2/India_83_83.png"
    assert os.path.exists(test_roi_path), f"Test image missing: {test_roi_path}"

    roi = cv2.imread(test_roi_path)
    assert roi is not None

    overlay, heatmap, err = explainer.explain(roi, target_class_anemia=True)
    assert err is None
    assert overlay is not None
    assert heatmap is not None
    assert overlay.shape == roi.shape
    assert np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0


def test_full_image_upload_with_gradcam(client):
    """Verify end-to-end upload with segmentation, classification, and Grad-CAM generation."""
    # Create or use a valid eye image
    test_roi_path = "data/processed/roi_predicted_v2/India_83_83.png"
    roi = cv2.imread(test_roi_path)

    # Embed ROI into a simulated eye canvas (256x256)
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    canvas[20:244, 20:244] = cv2.resize(roi, (224, 224))
    _, encoded_img = cv2.imencode(".png", canvas)

    data = {
        "eye_image": (io.BytesIO(encoded_img.tobytes()), "test_eye.png"),
        "sex": "male",
        "hb": "14.2",
    }

    response = client.post("/", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    html = response.data.decode("utf-8")

    # Confirm visual explainability and results
    assert "Assessment Results &amp; Explainability" in html or "Assessment Results" in html
    assert "Grad-CAM visualizes image regions" in html
    assert "data:image/png;base64," in html
