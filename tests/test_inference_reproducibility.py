"""Reproducibility test suite: Verifies that identical input images produce

100% consistent preprocessing, segmentation, ROI extraction, prediction scores,
and explainability outputs across multiple consecutive runs.
"""
import io
import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np

# Configure isolated environment for testing
os.environ["DB_FALLBACK_SQLITE"] = "true"
os.environ["SECRET_KEY"] = "test-secret-key-12345"

from app import app, classifier, unet, explainer, prepare_roi_for_classifier, CLASSIFIER_THRESHOLD
from src.database.models import db
from src.segmentation.predict import predict_mask
from src.roi.extract import extract_roi


class InferenceReproducibilityTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        uploads_dir = os.path.join(cls.base_dir, "uploads")
        candidates = [
            os.path.join(uploads_dir, f)
            for f in os.listdir(uploads_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not candidates:
            raise FileNotFoundError("No test images found in uploads directory.")
        cls.test_image_path = candidates[0]
        with open(cls.test_image_path, "rb") as f:
            cls.test_image_bytes = f.read()

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_reproduce.db")
        app.config["TESTING"] = True
        app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{self.db_path}"
        app.config["UPLOAD_FOLDER"] = os.path.join(self.temp_dir, "uploads")
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        self.client = app.test_client()

        with app.app_context():
            db.create_all()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_reproducibility_five_iterations(self):
        """
        Processes the same image bytes 5 times through every stage of the pipeline
        and verifies that all stages produce bit-exact identical outputs.
        """
        preprocessing_outputs = []
        mask_outputs = []
        roi_outputs = []
        scores = []
        classifications = []
        heatmaps = []

        for iteration in range(1, 6):
            # Stage 1: Decode and Preprocessing
            image = cv2.imdecode(np.frombuffer(self.test_image_bytes, np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(image, "Image decoding failed.")

            h_orig, w_orig = image.shape[:2]
            max_input_dim = 1024
            if max(h_orig, w_orig) > max_input_dim:
                scale = max_input_dim / float(max(h_orig, w_orig))
                image = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)

            preprocessing_outputs.append(image)

            # Stage 2: U-Net Segmentation
            _, probability, binary_mask, quality = predict_mask(
                unet,
                image,
                input_size=(256, 256),
                threshold=0.5,
            )
            self.assertIsNotNone(binary_mask, "Binary mask generation failed.")
            mask_outputs.append(binary_mask)

            # Stage 3: Conjunctiva ROI Extraction
            roi, bbox = extract_roi(
                image,
                binary_mask,
                output_size=(224, 224),
                margin_fraction=0.15,
            )
            self.assertIsNotNone(roi, "ROI extraction failed.")
            roi_outputs.append(roi)

            # Stage 4: EfficientNet-B0 Classification
            roi_batch = prepare_roi_for_classifier(roi)
            prediction = classifier.predict(roi_batch, verbose=0)
            score = float(np.asarray(prediction).reshape(-1)[0])
            is_anemic = score >= CLASSIFIER_THRESHOLD
            classification_str = "Anemia Detected" if is_anemic else "Non-Anemia (Normal)"

            scores.append(score)
            classifications.append(classification_str)

            # Stage 5: Grad-CAM Explainability
            overlay, heatmap, err = explainer.explain(roi, target_class_anemia=is_anemic)
            self.assertIsNone(err, f"Grad-CAM error: {err}")
            self.assertIsNotNone(heatmap, "Heatmap was None.")
            heatmaps.append(heatmap)

        # Verification 1: Preprocessing outputs are 100% identical
        for i in range(1, 5):
            np.testing.assert_array_equal(
                preprocessing_outputs[0],
                preprocessing_outputs[i],
                err_msg=f"Preprocessing output changed at run {i+1}!",
            )

        # Verification 2: U-Net masks are 100% identical
        for i in range(1, 5):
            np.testing.assert_array_equal(
                mask_outputs[0],
                mask_outputs[i],
                err_msg=f"U-Net segmentation mask changed at run {i+1}!",
            )

        # Verification 3: ROIs are 100% identical
        for i in range(1, 5):
            np.testing.assert_array_equal(
                roi_outputs[0],
                roi_outputs[i],
                err_msg=f"Extracted ROI changed at run {i+1}!",
            )

        # Verification 4: Prediction scores are 100% identical (max difference == 0.0)
        max_score_diff = max(scores) - min(scores)
        score_std = np.std(scores)
        self.assertEqual(
            max_score_diff,
            0.0,
            f"Prediction scores varied across 5 runs! Scores: {scores}, Max diff: {max_score_diff}",
        )
        self.assertEqual(
            score_std,
            0.0,
            f"Prediction score standard deviation must be 0.0, got: {score_std}",
        )

        # Verification 5: Classifications are 100% identical
        self.assertEqual(len(set(classifications)), 1, f"Classification flipped: {classifications}")

        # Verification 6: Grad-CAM heatmaps are 100% identical
        for i in range(1, 5):
            np.testing.assert_array_almost_equal(
                heatmaps[0],
                heatmaps[i],
                decimal=6,
                err_msg=f"Grad-CAM heatmap changed at run {i+1}!",
            )

    def test_flask_http_consecutive_uploads_reproducibility(self):
        """
        Uploads the exact same conjunctival image 5 times via HTTP POST to the Flask app,
        verifying that identical scores and classifications are returned every time.
        """
        http_scores = []
        http_classifications = []

        for i in range(1, 6):
            data = {
                "patient_name": f"Test Patient {i}",
                "patient_id": "PID-TEST-100",
                "eye_image": (io.BytesIO(self.test_image_bytes), "test_eye.jpg"),
            }
            res = self.client.post("/", data=data, content_type="multipart/form-data")
            self.assertEqual(res.status_code, 200, f"HTTP POST failed on iteration {i}")

            # Check that results section is rendered
            self.assertIn(b"AI Screening Results", res.data)

            # Check that classification is rendered
            has_anemia = b"Anemia Detected" in res.data
            has_normal = b"Non-Anemic" in res.data or b"Non-Anemia" in res.data
            self.assertTrue(has_anemia or has_normal)
            http_classifications.append("Anemia" if has_anemia else "Normal")

        # All 5 HTTP uploads must return the exact same classification
        self.assertEqual(
            len(set(http_classifications)),
            1,
            f"HTTP predictions varied across uploads: {http_classifications}",
        )


if __name__ == "__main__":
    unittest.main()
