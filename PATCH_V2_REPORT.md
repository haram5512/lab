# Adversarial patch pool v2 report

The original pool remains preserved as v1 under `artifacts/adversarial_patches/`.
The v2 trial is stored separately under `artifacts/adversarial_patches/v2/`.

v2 used 200 COCO train2017 source images, 500 optimization steps, shared
multi-image optimization, and wider scale/translation/rotation/appearance EOT.
Held-out validation used 20 COCO validation images per patch.

The trial was rejected as the main pool: v2 did not produce a consistent
held-out score reduction. Several patches increased the measured raw/NMS score,
and detection retention stayed around 0.90--0.95. The v1 pool remains the
strongest validated pool and is not overwritten. Full 100-epoch training is
therefore intentionally not started.

The generator now samples one source mini-batch per optimization step and
releases each computation graph, avoiding CUDA memory growth for larger source
sets. Full homography remains a future physical-robustness extension.
