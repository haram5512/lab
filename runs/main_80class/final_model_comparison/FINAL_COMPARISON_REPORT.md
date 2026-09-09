# Final Model Comparison

Final verdict: **NO-GO**.

The machine-readable tables in this directory are the authoritative record. Evaluation uses COCO val2017 person-containing images (11,004 person GT), 640 letterbox/inverse-letterbox, COCOeval, confidence 0.25 for operating-point Recall/Precision/F1, and one-to-one greedy person matching.

Proposed v2 training stopped manually after epoch 60 because Seen AP did not meaningfully exceed its previous best.
