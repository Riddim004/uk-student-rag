# Relevance gate calibration

Commit: `7b88dec` | rerank: `True` | candidate_k: 50 | rerank_k: 15

The gate answers only when the strongest underlying score clears a
threshold. Thresholds are selected on the **dev** split; the **test**
split is a held-out check and is not used for selection.

## Score distributions

| split | label | field | n | min | median | max |
|---|---|---|---|---|---|---|
| dev | answerable | max_dense | 21 | 0.806 | 0.891 | 0.922 |
| dev | answerable | max_sparse | 21 | 0.000 | 16.872 | 30.887 |
| dev | answerable | max_rerank | 21 | -2.890 | 4.293 | 10.763 |
| dev | unanswerable | max_dense | 4 | 0.847 | 0.864 | 0.874 |
| dev | unanswerable | max_sparse | 4 | 7.647 | 8.757 | 11.612 |
| dev | unanswerable | max_rerank | 4 | -4.156 | -3.733 | 5.100 |
| test | answerable | max_dense | 21 | 0.824 | 0.885 | 0.928 |
| test | answerable | max_sparse | 21 | 0.000 | 15.146 | 24.831 |
| test | answerable | max_rerank | 21 | -2.251 | 3.150 | 10.794 |
| test | unanswerable | max_dense | 4 | 0.785 | 0.862 | 0.891 |
| test | unanswerable | max_sparse | 4 | 8.630 | 10.367 | 20.804 |
| test | unanswerable | max_rerank | 4 | -5.365 | 0.248 | 4.066 |

## Dense threshold sweep (dev)

| threshold | unanswerable refused | answerable over-refused | balanced accuracy |
|---|---|---|---|
| 0.700 | 0.000 | 0.000 | 0.500 |
| 0.705 | 0.000 | 0.000 | 0.500 |
| 0.710 | 0.000 | 0.000 | 0.500 |
| 0.715 | 0.000 | 0.000 | 0.500 |
| 0.720 | 0.000 | 0.000 | 0.500 |
| 0.725 | 0.000 | 0.000 | 0.500 |
| 0.730 | 0.000 | 0.000 | 0.500 |
| 0.735 | 0.000 | 0.000 | 0.500 |
| 0.740 | 0.000 | 0.000 | 0.500 |
| 0.745 | 0.000 | 0.000 | 0.500 |
| 0.750 | 0.000 | 0.000 | 0.500 |
| 0.755 | 0.000 | 0.000 | 0.500 |

## Sparse threshold sweep (dev)

| threshold | unanswerable refused | answerable over-refused | balanced accuracy |
|---|---|---|---|
| 12.000 | 1.000 | 0.333 | 0.833 |
| 15.000 | 1.000 | 0.381 | 0.810 |
| 10.000 | 0.750 | 0.286 | 0.732 |
| 8.000 | 0.500 | 0.190 | 0.655 |
| 0.000 | 0.000 | 0.143 | 0.429 |
| 1.000 | 0.000 | 0.143 | 0.429 |
| 2.000 | 0.000 | 0.143 | 0.429 |
| 3.000 | 0.000 | 0.143 | 0.429 |
| 4.000 | 0.000 | 0.143 | 0.429 |
| 5.000 | 0.000 | 0.143 | 0.429 |
| 6.000 | 0.000 | 0.190 | 0.405 |

## Rerank threshold sweep (dev)

| threshold | unanswerable refused | answerable over-refused | balanced accuracy |
|---|---|---|---|
| -5.000 | 0.000 | 0.000 | 0.500 |
| -4.950 | 0.000 | 0.000 | 0.500 |
| -4.900 | 0.000 | 0.000 | 0.500 |
| -4.850 | 0.000 | 0.000 | 0.500 |
| -4.800 | 0.000 | 0.000 | 0.500 |
| -4.750 | 0.000 | 0.000 | 0.500 |
| -4.700 | 0.000 | 0.000 | 0.500 |
| -4.650 | 0.000 | 0.000 | 0.500 |
| -4.600 | 0.000 | 0.000 | 0.500 |
| -4.550 | 0.000 | 0.000 | 0.500 |
| -4.500 | 0.000 | 0.000 | 0.500 |
| -4.450 | 0.000 | 0.000 | 0.500 |

## Selected thresholds

| rule | threshold | dev unanswerable refused | dev answerable over-refused | dev balanced accuracy |
|---|---|---|---|---|
| dense | 0.875 | 1.000 | 0.381 | 0.810 |
| sparse | 12.000 | 1.000 | 0.333 | 0.833 |
| rerank | -3.000 | 0.750 | 0.000 | 0.875 |

The rerank threshold is the highest dev-optimal threshold minus a 0.1 margin, so it does not sit exactly on the lowest-scoring answerable query. The dense/sparse scalar thresholds are reported as a negative result: they need 33–38% over-refusal to reach the same dev refusal rate.

## Held-out check (test split, thresholds frozen)

| rule | unanswerable refused | answerable over-refused | gate accuracy | balanced accuracy |
|---|---|---|---|---|
| rerank | 0.500 | 0.000 | 0.920 | 0.750 |

Caveat: the test split has only 4 unanswerable and 21 answerable queries, so these intervals are wide.
