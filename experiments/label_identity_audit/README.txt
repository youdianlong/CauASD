CauASD action-identity audit
============================

This auxiliary audit examines action-identity preservation under fixed-50-frame
temporal intervention. It is not used for training or recognition evaluation.

Released files
--------------
Each dataset directory contains:
  annotation_sheet.csv    completed anonymous ratings (annotator_1/annotator_2)
  pair_key.csv            pair ID, original sample ID, split, class, and factor
  pairs/                  original/transformed paired videos
  boundary_statistics.csv temporal coverage and endpoint statistics
NTU-60 additionally contains audit_pairs.npz with the selected-pair data.
Ratings have been transcribed from the annotators' confirmed final records.

Judgment task and scoring
-------------------------
Original is shown on the left and Transformed on the right. The action
category is shown; the factor is omitted from the annotation sheet and stored
in pair_key.csv. Each annotator independently judges whether the transformed
motion retains the action identity of the original:
  Yes       action-identity preservation can be confirmed.
  Uncertain preservation cannot be confidently confirmed from the display.

A pair is confirmed only when BOTH annotators select Yes:
  Confirmation rate = number of (Yes, Yes) pairs / total evaluated pairs.
Every other completed combination is unconfirmed and remains in the denominator.
Uncertain does not establish a changed action label. A blank is a missing rating,
not Uncertain. The released completed sheets contain no blanks or No ratings.

Sample selection
----------------
The generation scripts sample original unseen-test sequences within each split:
  NTU-60: 50 originals, with 17/17/16 from splits 1/2/3, using approximately
          class-balanced sampling within each split.
  NTU-120: one original per unseen class per split, giving 10/10/10 originals
           from splits 4/5/6 (30 originals).
The generation scripts default to seed 2026; pair_key.csv identifies the exact
released sample selections. Each original produces six variants at factors
0.5, 0.75, 1.25, 1.5, 1.75, and 2.0, using tau[t] = clip(t / factor, 0, 49).
Thus 300/180 pairs are repeated transformations of 50/30 originals, not 300/180
independent original sequences.

Results: pairs confirmed by both annotators
------------------------------------------
Factor     NTU-60             NTU-120
0.50       50/50 (100.00%)    30/30 (100.00%)
0.75       50/50 (100.00%)    30/30 (100.00%)
1.25       50/50 (100.00%)    30/30 (100.00%)
1.50       50/50 (100.00%)    30/30 (100.00%)
1.75       49/50 (98.00%)     28/30 (93.33%)
2.00       45/50 (90.00%)     25/30 (83.33%)
Overall    294/300 (98.00%)   173/180 (96.11%)

This is a contextual paired audit with category information, not a blinded
recognition test or a guarantee for every transformed sample. At extreme
factors, partial trajectory coverage or endpoint repetition can limit the
available identity cues. Unconfirmed cases are reported, not relabeled.

Generation scripts
------------------
generate_label_identity_audit.py       NTU-60
generate_ntu120_label_identity_audit.py NTU-120
The scripts generate blank forms for NEW audits; they do not recreate human
ratings. Use a new output directory rather than overwriting the completed audit.
No checkpoint is required.
