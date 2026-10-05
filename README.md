# LinerScan

3D laser-triangulation bore scanner for the cylinder liners of large two-stroke marine diesel engines (400–980 mm
bore), for UMA Marine. The device lies on the piston crown, goes in through a scavenge port, and scans the whole
liner wall in **one turning-gear stroke** from BDC to a planned stop angle. It runs without a cable, stores the data
on board and sends status over BLE. Processing runs after retrieval.

This repository holds the **complete processing chain, a physics-level simulator of the device, and the on-board
logic** (controller, crush safety) written against driver interfaces. Real drivers can be added later.
Everything here runs in simulation.

```
pip install -e .[dev]
pytest -q                         # unit + end-to-end simulation tests
linerscan plan --bore 980 --stroke 2660 --conrod 2885
linerscan simulate --preset L --quick --out out/sim_L
linerscan process out/sim_L/raw.npz --cal out/sim_L/cal.json --job out/sim_L/job.json \
                  --engine out/sim_L/engine.json --pdf out/sim_L/report.pdf
linerscan example                 # regenerates examples/LinerScan_example_report.pdf
linerscan validate                # simulation campaign → docs/VALIDATION.md
```

| Path | What |
|---|---|
| `linerscan/optics.py` | Row models z = a/(u−u0)+b, distortion and cubic variants, VarPro + LM, LOO model choice, surface ("plane") model, thermal model |
| `linerscan/geometry.py` | r, α, height from z; ring calibration of R0 with a prior on h |
| `linerscan/fitting.py` | Kåsa/LM circle, cylinder, shape-aware axis fit, Fourier D(α)=ρ(α)+ρ(α+π), ellipse, MAD outliers |
| `linerscan/pipeline.py` | raw scan → points → height reference → bands → results, yaw from the port pattern |
| `linerscan/report.py` | PDF survey report |
| `linerscan/kinematics.py` | slider-crank, planned stop angle, scan planning (sweep rate, alarm levels) |
| `linerscan/device/` | driver interfaces, scan controller, ToF crush-safety monitor, sweep profile |
| `linerscan/sim/` | liner/ring truth, camera/encoder/thermal/ToF models, world, factory calibration rig, validation |
| `docs/` | DECISIONS, BOM, VALIDATION, HANDOFF |
| `examples/` | example (simulated) report, band CSV, BLE log |
