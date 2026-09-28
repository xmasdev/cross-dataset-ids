# DA-MFS — 3x3 cross-dataset matrices (rows = trained on, cols = tested on)


## Accuracy


**Raw, all features (baseline / before)**

| trained \ tested | CIC | UNSW | TON |
|---|---|---|---|
| CIC | **0.982** | 0.361 | 0.262 |
| UNSW | 0.263 | **0.946** | 0.660 |
| TON | 0.404 | 0.286 | **0.972** |

**Aligned, all features**

| trained \ tested | CIC | UNSW | TON |
|---|---|---|---|
| CIC | **0.982** | 0.348 | 0.429 |
| UNSW | 0.340 | **0.946** | 0.752 |
| TON | 0.361 | 0.636 | **0.972** |

**DA-MFS (aligned + optimised subset / after)**

| trained \ tested | CIC | UNSW | TON |
|---|---|---|---|
| CIC | **0.980** | 0.374 | 0.506 |
| UNSW | 0.431 | **0.946** | 0.590 |
| TON | 0.458 | 0.691 | **0.947** |

## Balanced accuracy


**Raw, all features (baseline / before)**

| trained \ tested | CIC | UNSW | TON |
|---|---|---|---|
| CIC | **0.984** | 0.500 | 0.516 |
| UNSW | 0.490 | **0.952** | 0.442 |
| TON | 0.561 | 0.281 | **0.943** |

**Aligned, all features**

| trained \ tested | CIC | UNSW | TON |
|---|---|---|---|
| CIC | **0.984** | 0.477 | 0.586 |
| UNSW | 0.578 | **0.952** | 0.499 |
| TON | 0.519 | 0.519 | **0.943** |

**DA-MFS (aligned + optimised subset / after)**

| trained \ tested | CIC | UNSW | TON |
|---|---|---|---|
| CIC | **0.983** | 0.506 | 0.636 |
| UNSW | 0.629 | **0.952** | 0.478 |
| TON | 0.547 | 0.619 | **0.926** |

## Macro-F1


**Raw, all features (baseline / before)**

| trained \ tested | CIC | UNSW | TON |
|---|---|---|---|
| CIC | **0.972** | 0.265 | 0.227 |
| UNSW | 0.258 | **0.942** | 0.416 |
| TON | 0.400 | 0.279 | **0.960** |

**Aligned, all features**

| trained \ tested | CIC | UNSW | TON |
|---|---|---|---|
| CIC | **0.972** | 0.268 | 0.429 |
| UNSW | 0.339 | **0.942** | 0.446 |
| TON | 0.359 | 0.466 | **0.960** |

**DA-MFS (aligned + optimised subset / after)**

| trained \ tested | CIC | UNSW | TON |
|---|---|---|---|
| CIC | **0.969** | 0.295 | 0.502 |
| UNSW | 0.430 | **0.943** | 0.477 |
| TON | 0.439 | 0.620 | **0.926** |

## Attack recall


**Raw, all features (baseline / before)**

| trained \ tested | CIC | UNSW | TON |
|---|---|---|---|
| CIC | **0.988** | 0.000 | 0.033 |
| UNSW | 0.866 | **0.930** | 0.856 |
| TON | 0.820 | 0.299 | **0.998** |

**Aligned, all features**

| trained \ tested | CIC | UNSW | TON |
|---|---|---|---|
| CIC | **0.989** | 0.013 | 0.287 |
| UNSW | 0.971 | **0.930** | 0.980 |
| TON | 0.779 | 0.938 | **0.998** |

**DA-MFS (aligned + optimised subset / after)**

| trained \ tested | CIC | UNSW | TON |
|---|---|---|---|
| CIC | **0.988** | 0.031 | 0.388 |
| UNSW | 0.956 | **0.932** | 0.692 |
| TON | 0.693 | 0.879 | **0.965** |
