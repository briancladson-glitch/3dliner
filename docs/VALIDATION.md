# Simulation validation

Measured − ground truth for every 30 mm profile band (port-zone and steep-gradient bands listed separately).  All values in mm.  Truth = order-8 Fourier projection of the true band-mean profile (what an ideal instrument would report).  Regenerate with `python -m linerscan.sim.campaign`.

| case | bands | max\|err\| D_FA | D_PS | ovality | D_max | D_min | RMS D_FA | RMS D_PS | yaw err ° | height err max | model |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S bore 450, liner 62 °C, seed 0 | 54 | 0.006 | 0.013 | 0.014 | 0.005 | 0.011 | 0.0023 | 0.0074 | -0.25 | 2.8 | distortion/surface |
| M bore 650, liner 62 °C, seed 0 | 75 | 0.011 | 0.007 | 0.015 | 0.011 | 0.009 | 0.0055 | 0.0032 | -0.32 | 4.2 | hyperbolic/surface |
| L bore 980, liner 62 °C, seed 0 | 78 | 0.017 | 0.013 | 0.016 | 0.013 | 0.016 | 0.0061 | 0.0045 | -0.20 | 1.2 | hyperbolic/surface |
| L bore 980, liner 62 °C, seed 1 | 78 | 0.017 | 0.014 | 0.017 | 0.015 | 0.014 | 0.0062 | 0.0067 | -0.20 | 1.3 | hyperbolic/surface |
| L bore 980, liner 80 °C, seed 0 | 78 | 0.020 | 0.013 | 0.015 | 0.015 | 0.019 | 0.0080 | 0.0056 | -0.20 | 1.1 | hyperbolic/surface |
| L bore 980, liner 80 °C, seed 0, NO thermal compensation | 78 | 2.697 | 2.693 | 0.023 | 2.699 | 2.696 | 2.6707 | 2.6665 | -0.20 | 1.3 | hyperbolic/surface |

## Special bands

| case | port-zone bands: max\|err\| D_FA / D_PS | steep-gradient bands: n, max\|err\| D_mean |
|---|---|---|
| S bore 450, liner 62 °C, seed 0 | 0.002 / 0.009 | 1, 0.052 |
| M bore 650, liner 62 °C, seed 0 | 0.009 / 0.005 | 2, 0.070 |
| L bore 980, liner 62 °C, seed 0 | 0.011 / 0.005 | 2, 0.045 |
| L bore 980, liner 62 °C, seed 1 | 0.006 / 0.013 | 2, 0.060 |
| L bore 980, liner 80 °C, seed 0 | 0.010 / 0.007 | 2, 0.039 |
| L bore 980, liner 80 °C, seed 0, NO thermal compensation | 2.644 / 2.639 | 2, 2.676 |

## Standard measuring positions (VAK-style table)

| case | positions | max\|err\| D_FA | D_PS | ovality | D_max | D_min |
|---|---|---|---|---|---|---|
| S bore 450, liner 62 °C, seed 0 | 14 | 0.003 | 0.010 | 0.011 | 0.005 | 0.011 |
| M bore 650, liner 62 °C, seed 0 | 15 | 0.008 | 0.006 | 0.009 | 0.009 | 0.011 |
| L bore 980, liner 62 °C, seed 0 | 15 | 0.013 | 0.009 | 0.013 | 0.008 | 0.012 |
| L bore 980, liner 62 °C, seed 1 | 15 | 0.010 | 0.010 | 0.014 | 0.008 | 0.012 |
| L bore 980, liner 80 °C, seed 0 | 15 | 0.016 | 0.012 | 0.013 | 0.011 | 0.014 |
| L bore 980, liner 80 °C, seed 0, NO thermal compensation | 15 | 2.692 | 2.693 | 0.020 | 2.693 | 2.695 |

Points, outliers and run time:

- S bore 450, liner 62 °C, seed 0: 59741 frames, 2,259,320 points, 4,820 marked outliers, device 60.3 °C, tilt true [0.35, -0.2] est [0.341, -0.220], 65 s
- M bore 650, liner 62 °C, seed 0: 60701 frames, 2,274,161 points, 4,548 marked outliers, device 60.3 °C, tilt true [0.35, -0.2] est [0.347, -0.227], 69 s
- L bore 980, liner 62 °C, seed 0: 60741 frames, 2,260,803 points, 4,469 marked outliers, device 60.3 °C, tilt true [0.35, -0.2] est [0.330, -0.219], 73 s
- L bore 980, liner 62 °C, seed 1: 60721 frames, 2,260,723 points, 4,637 marked outliers, device 60.3 °C, tilt true [0.35, -0.2] est [0.329, -0.221], 77 s
- L bore 980, liner 80 °C, seed 0: 60741 frames, 2,260,675 points, 4,468 marked outliers, device 78.3 °C, tilt true [0.35, -0.2] est [0.330, -0.220], 50 s
- L bore 980, liner 80 °C, seed 0, NO thermal compensation: 60741 frames, 2,260,675 points, 4,468 marked outliers, device 78.3 °C, tilt true [0.35, -0.2] est [0.329, -0.219], 50 s
