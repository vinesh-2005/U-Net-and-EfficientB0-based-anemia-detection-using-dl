"""Grad-CAM (Gradient-weighted Class Activation Mapping) for EfficientNetB0."""
import logging
from typing import Optional, Tuple
import cv2
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class GradCAMExplainer:
    """
    Computes genuine Grad-CAM heatmaps for the trained EfficientNetB0 classifier.
    Extracts gradients from the last convolutional activation layer (top_activation)
    with respect to the class prediction score.
    """

    def __init__(self, classifier_model: tf.keras.Model, conv_layer_name: str = "top_activation"):
        self.model = classifier_model
        self.conv_layer_name = conv_layer_name
        self.feature_model = None
        self.head_model = None
        self._build_gradcam_models()

    def _build_gradcam_models(self):
        """Constructs gradient-trackable submodels from the loaded model architecture."""
        try:
            # Locate the EfficientNet backbone functional model
            backbone = None
            for layer in self.model.layers:
                if "efficientnet" in layer.name.lower():
                    backbone = layer
                    break

            if backbone is None:
                logger.error("Could not locate EfficientNet backbone in classifier model.")
                return

            target_layer = backbone.get_layer(self.conv_layer_name)

            # Feature model extracts convolutional feature maps and backbone pooled output
            self.feature_model = tf.keras.Model(
                inputs=backbone.inputs,
                outputs=[target_layer.output, backbone.output],
                name="gradcam_feature_model",
            )

            # Head model takes backbone output and computes prediction
            head_input = tf.keras.Input(shape=backbone.output.shape[1:])
            x = head_input
            head_layer_names = [
                "global_average_pooling2d",
                "batch_normalization",
                "dropout",
                "dense",
                "dropout_1",
                "dense_1",
            ]
            for name in head_layer_names:
                try:
                    layer = self.model.get_layer(name)
                    x = layer(x, training=False)
                except ValueError:
                    logger.warning(f"Head layer {name} not found in model.")

            self.head_model = tf.keras.Model(
                inputs=head_input,
                outputs=x,
                name="gradcam_head_model",
            )
            logger.info("GradCAMExplainer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize GradCAM submodels: {e}")
            self.feature_model = None
            self.head_model = None

    def explain(
        self,
        roi_bgr: np.ndarray,
        target_class_anemia: Optional[bool] = None,
        alpha: float = 0.6,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """
        Computes the Grad-CAM heatmap and overlays it onto the extracted ROI.

        Args:
            roi_bgr: BGR ROI image (e.g. 224x224x3).
            target_class_anemia: If True, explain positive (Anemic) score.
                                 If False, explain negative (Normal) score.
                                 If None, explain the model's predicted class.
            alpha: Weight for the original image in the alpha blend (1-alpha for heatmap).

        Returns:
            Tuple of (overlay_bgr, heatmap_normalized_2d, error_message).
        """
        if self.feature_model is None or self.head_model is None:
            return None, None, "Grad-CAM models could not be initialized from the classifier."

        if roi_bgr is None or roi_bgr.size == 0:
            return None, None, "Input ROI is empty or invalid."

        try:
            h, w = roi_bgr.shape[:2]

            # Preprocess matching the classifier's expected input (RGB float32, range [0, 255])
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            roi_resized = cv2.resize(roi_rgb, (224, 224))
            img_tensor = np.expand_dims(roi_resized.astype(np.float32), axis=0)

            # GradientTape to track gradients of score with respect to convolutional feature maps
            with tf.GradientTape() as tape:
                conv_outputs, backbone_out = self.feature_model(img_tensor, training=False)
                tape.watch(conv_outputs)
                preds = self.head_model(backbone_out, training=False)
                score = float(preds[0, 0])

                if target_class_anemia is None:
                    # Target the predicted class direction
                    target_class_anemia = score >= 0.3837

                if target_class_anemia:
                    loss = preds[:, 0]
                else:
                    loss = 1.0 - preds[:, 0]

            # Compute gradients of target score with respect to feature maps
            grads = tape.gradient(loss, conv_outputs)
            if grads is None:
                return None, None, "Could not compute gradients for the target layer."

            # Global average pooling of gradients: channel weights alpha_k
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Weighted combination of feature maps
            conv_outputs_0 = conv_outputs[0]
            heatmap = conv_outputs_0 @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # Apply ReLU to keep only features that positively influence the target class
            heatmap = tf.maximum(heatmap, 0.0)
            max_val = tf.math.reduce_max(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val

            heatmap_np = heatmap.numpy()

            # Resize heatmap to match ROI dimensions
            heatmap_resized = cv2.resize(heatmap_np, (w, h), interpolation=cv2.INTER_LINEAR)
            heatmap_uint8 = np.uint8(255 * heatmap_resized)

            # Generate JET colormap for visualization
            color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # Alpha blend overlay with original ROI
            overlay_bgr = cv2.addWeighted(roi_bgr, alpha, color_heatmap, 1.0 - alpha, 0)

            return overlay_bgr, heatmap_resized, None

        except Exception as e:
            logger.exception("Error generating Grad-CAM:")
            return None, None, f"Grad-CAM generation error: {str(e)}"
