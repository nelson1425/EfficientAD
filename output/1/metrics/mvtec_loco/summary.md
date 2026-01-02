# MVTec LOCO 评测结果汇总

- metrics_dir: `output/1/metrics/mvtec_loco`
- objects: 5
- format: percent(0-100)

## Image-level (AUC ROC)

| object | mean | logical_anomalies | structural_anomalies |
| --- | ---: | ---: | ---: |
| breakfast_box | 86.43 | 85.78 | 87.09 |
| juice_bottle | 99.91 | 99.95 | 99.88 |
| pushpins | 96.97 | 97.44 | 96.50 |
| screw_bag | 70.18 | 51.14 | 89.22 |
| splicing_connectors | 96.31 | 94.13 | 98.48 |
| avg | 89.96 | 85.69 | 94.23 |

## Localization (AU-sPRO mean)

| object | @0.01 | @0.05 | @0.1 | @0.3 | @1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| breakfast_box | 41.61 | 64.87 | 71.60 | 76.93 | 89.91 |
| juice_bottle | 71.88 | 93.45 | 96.64 | 98.82 | 99.64 |
| pushpins | 68.75 | 85.61 | 89.41 | 93.17 | 96.90 |
| screw_bag | 36.68 | 66.44 | 76.93 | 84.21 | 93.45 |
| splicing_connectors | 70.81 | 89.11 | 93.05 | 95.98 | 98.48 |
| avg | 57.95 | 79.90 | 85.53 | 89.82 | 95.68 |

## Localization (AU-sPRO logical_anomalies)

| object | @0.01 | @0.05 | @0.1 | @0.3 | @1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| breakfast_box | 31.52 | 56.11 | 63.40 | 68.82 | 85.78 |
| juice_bottle | 75.34 | 94.49 | 97.23 | 99.07 | 99.72 |
| pushpins | 80.19 | 95.68 | 97.79 | 99.26 | 99.78 |
| screw_bag | 14.24 | 50.49 | 65.54 | 75.81 | 89.86 |
| splicing_connectors | 67.55 | 87.14 | 92.07 | 95.75 | 98.30 |
| avg | 53.77 | 76.78 | 83.21 | 87.74 | 94.69 |

## Localization (AU-sPRO structural_anomalies)

| object | @0.01 | @0.05 | @0.1 | @0.3 | @1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| breakfast_box | 51.70 | 73.63 | 79.80 | 85.04 | 94.03 |
| juice_bottle | 68.43 | 92.40 | 96.05 | 98.57 | 99.56 |
| pushpins | 57.31 | 75.54 | 81.03 | 87.08 | 94.02 |
| screw_bag | 59.13 | 82.40 | 88.31 | 92.60 | 97.05 |
| splicing_connectors | 74.06 | 91.08 | 94.04 | 96.20 | 98.67 |
| avg | 62.13 | 83.01 | 87.85 | 91.90 | 96.67 |
