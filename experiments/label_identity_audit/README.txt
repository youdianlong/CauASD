CauASD action-identity audit
============================

This directory contains the optional qualitative audit of label preservation
under the fixed-window temporal intervention. It is not used by model
training or standard recognition testing.

Directory layout
----------------
ntu60/
  pairs/                 paired original/transformed videos (300 MP4 files)
  pair_key.csv           factor and class metadata for each pair
  annotation_sheet.csv   sheet for independent Yes/No/Uncertain judgments
  boundary_statistics.csv coverage and endpoint statistics
  audit_pairs.npz        reproducible selected-pair data (optional)

ntu120/
  pairs/                 paired original/transformed videos (180 MP4 files)
  pair_key.csv
  annotation_sheet.csv
  boundary_statistics.csv

The left panel of every video is Original and the right panel is Transformed.
The action category is displayed for inspection. The speed factor is recorded
in pair_key.csv rather than in the annotation sheet. Fill annotator_1 and
annotator_2 with Yes, No, or Uncertain, and compute retention as:

Retention = Yes / (Yes + No + Uncertain)

Generation scripts
------------------
generate_label_identity_audit.py       NTU-60 audit generation
generate_ntu120_label_identity_audit.py NTU-120 audit generation

The audit uses the legacy fixed-50-frame temporal intervention and the factors
0.5, 0.75, 1.25, 1.5, 1.75, and 2.0. No checkpoint is required.
