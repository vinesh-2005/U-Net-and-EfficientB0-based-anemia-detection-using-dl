# Anemia Prediction ML Architecture

This implementation follows `uploads/anaemia.md`: full RGB eye image -> verified conjunctiva U-Net -> deterministic mask quality checks and ROI extraction -> EfficientNetB0 transfer-learning classifier. The two trainable models are separate but are composed by `src/pipeline/infer.py`.

The repository contains no EYES-DEFY-ANEMIA dataset. It deliberately refuses to train or assign labels until the dataset's clinical table, patient-ID mapping, annotation semantics, and study anemia rule are verified. `_palpebral`, `_forniceal`, and `_forniceal_palpebral` are not presumed to be masks.

Ground-truth ROI experiments are diagnostic only. Predicted-mask ROIs are the primary end-to-end path.
