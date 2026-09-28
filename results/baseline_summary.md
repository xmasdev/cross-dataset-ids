### Same-dataset vs cross-dataset performance (full datasets)

| Model | Same-dataset Acc | Cross-dataset Acc | Same-dataset Bal-Acc | Cross-dataset Bal-Acc | Same-dataset F1 | Cross-dataset F1 |
|---|---|---|---|---|---|---|
| et | 0.9602 | 0.4027 | 0.9531 | 0.4672 | 0.9508 | 0.3444 |
| hgb | 0.9638 | 0.4179 | 0.9510 | 0.4941 | 0.9549 | 0.3799 |
| rf | 0.9666 | 0.3726 | 0.9597 | 0.4650 | 0.9581 | 0.3076 |
| vote | 0.9656 | 0.3968 | 0.9558 | 0.4624 | 0.9573 | 0.3453 |
| xgb | 0.9622 | 0.4150 | 0.9552 | 0.4760 | 0.9534 | 0.3597 |


**et — accuracy (rows = trained on, cols = tested on)**

| trained \ tested | CIC-IDS-2017 | UNSW-NB15 | TON_IoT |
|---|---|---|---|
| CIC-IDS-2017 | 0.9803 | 0.3610 | 0.4393 |
| UNSW-NB15 | 0.2802 | 0.9314 | 0.6869 |
| TON_IoT | 0.3757 | 0.2728 | 0.9690 |

**hgb — accuracy (rows = trained on, cols = tested on)**

| trained \ tested | CIC-IDS-2017 | UNSW-NB15 | TON_IoT |
|---|---|---|---|
| CIC-IDS-2017 | 0.9831 | 0.3591 | 0.4399 |
| UNSW-NB15 | 0.4383 | 0.9350 | 0.4633 |
| TON_IoT | 0.3360 | 0.4709 | 0.9732 |

**rf — accuracy (rows = trained on, cols = tested on)**

| trained \ tested | CIC-IDS-2017 | UNSW-NB15 | TON_IoT |
|---|---|---|---|
| CIC-IDS-2017 | 0.9817 | 0.3610 | 0.2618 |
| UNSW-NB15 | 0.2630 | 0.9457 | 0.6601 |
| TON_IoT | 0.4038 | 0.2858 | 0.9723 |

**vote — accuracy (rows = trained on, cols = tested on)**

| trained \ tested | CIC-IDS-2017 | UNSW-NB15 | TON_IoT |
|---|---|---|---|
| CIC-IDS-2017 | 0.9837 | 0.3609 | 0.4398 |
| UNSW-NB15 | 0.2765 | 0.9402 | 0.5289 |
| TON_IoT | 0.3359 | 0.4386 | 0.9731 |

**xgb — accuracy (rows = trained on, cols = tested on)**

| trained \ tested | CIC-IDS-2017 | UNSW-NB15 | TON_IoT |
|---|---|---|---|
| CIC-IDS-2017 | 0.9807 | 0.3610 | 0.4429 |
| UNSW-NB15 | 0.2874 | 0.9332 | 0.6515 |
| TON_IoT | 0.3533 | 0.3942 | 0.9727 |