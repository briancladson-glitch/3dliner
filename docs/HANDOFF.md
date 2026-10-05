# LinerScan — handoff

Status on 2026-10-05. Everything runs in simulation: there is no hardware, no engine data and no UMA measurements yet.

## What works — with proof

| Area | What | Proof |
|---|---|---|
| Triangulation | Per-row z = a/(u−u0)+b. VarPro 1-D search for u0, then LM. Distortion and cubic variants, a surface ("plane") model, and leave-one-out selection with the ≥ 20 % and ≥ 1 µm rule. Thermal drift model calibrated in a chamber. | `tests/test_optics_geometry.py`: exact recovery, correct model chosen for clean / distorted / odd data, thermal error < 0.01 mm at 80 °C |
| Head geometry | r, α_wall and height formulas; R0 from a ring by iteration; h with a 1 mm prior; arm clearance (no blind zones). | Known-answer tests; ring test recovers R0 to 2 µm |
| Band processing | Robust cylinder fit, then a **shape-aware** tilt fit (D4), circle (Kåsa → LM) about the band axis, order-8 Fourier with taper, D_FA / D_PS / ovality / D max / D min with angles / max−min, ellipse cross-check, MAD outliers marked and counted. | `tests/test_fitting.py` plus end-to-end tests |
| Height and orientation | Laser rangefinder + BDC / liner-top-edge anchors. FORE direction refined from the scavenge-port pattern. | E2E: height error < 3 mm; yaw error 0.2–0.3° from a placement that was 2.7° off |
| Device logic | Controller states (soak → armed → scan → stopped → done), BLE status with 20 % packet loss, quick-look diameters, laser interlock. | `tests/test_device.py` |
| Crush safety | Planned stop angle plus ToF WARN / ALARM / time-to-contact / FAULT, buzzer, independent of the radio. | Overshoot scenario with the **radio dead**: alarm, operator stops, more than 5 mm left; ToF failure → FAULT → stop |
| Report | 7-page PDF: summary, VAK-style table, profile, wall map, cross-sections, 3D view, QA, method, sign-off, full band appendix. | `examples/LinerScan_example_report.pdf` |

**Tests:** `pytest -q` → **{N_TESTS} passed** (about {T_TESTS} min; factory calibrations are cached per session). `ruff check .` is clean.

**Simulation error figures** (measured − truth, every 30 mm band outside the port zone and the ring-reversal ridge; full table in `docs/VALIDATION.md`):

{VAL_TABLE}

The spec is ±0.05 mm. The worst band in every case is below {WORST} mm, including a liner at **80 °C**. With thermal compensation switched off, the same 80 °C scan is out by **{NOCOMP} mm**, so the compensation is essential.

**Example report:** `examples/LinerScan_example_report.pdf`, a simulated 980-bore liner at 62 °C. The band data is in `examples/example_bands.csv` and the BLE status log in `examples/example_ble_status.jsonl`. Regenerate with `linerscan example`.

## What is left

1. **Real drivers.**
   - Camera SoM with on-board peak detection (port `peak.detect_peak` to C/NEON).
   - Encoder, VL53L8CX (ST ULD driver) and the laser distance module.
   - nRF52840 safety firmware (port `device/safety.py`).
   - BLE GATT profile.
   - Streaming the raw format to eMMC.

   The interfaces are in `device/drivers.py`; the controller does not change.
2. **Hardware prototype.** Mechanical fit in 20 mm, thermal behaviour of the battery and optics, laser class.
3. **Real calibration.** Rail and target rig, chamber campaign, master rings. Replace the simulated noise, distortion, drift and encoder values with measured ones, then re-run `linerscan validate`.
4. **Field trial.** Scan side by side with the VAK / LDM on a real cylinder. Set u_sys (currently 0.008 mm, simulation-derived) from that comparison.
5. **Not modelled yet:**
   - tangentially inclined port walls;
   - an anti-polishing ring step;
   - the cover bore above the liner top (the top-edge anchor assumes a clean edge);
   - oil film or carbon on the wall;
   - piston rocking within its clearance;
   - encoder self-calibration (harmonic errors are only simulated).
6. **Speed.** About 45 s to process one stroke on a laptop. That is fine, and could be parallelised per band.

## Open questions for Brian (R = changes how the report states a condition)

1. **R — Temperature reference.** Should diameters be stated as measured at liner temperature (current draft), or corrected to 20 °C? At 62 °C a 980 bore is about 0.45 mm larger. At 80 °C it is about 0.65 mm larger.
2. **R — Limits and wording.** Which wear and ovality limits apply per engine type? Is "diameter increase" D max − nominal (draft), D mean − nominal, or per direction (FA / PS)? Are the "WITHIN / APPROACHING (≥ 80 %) / EXCEEDS" wording and threshold OK? The limits are placeholders now.
3. **R — Angle convention.** Draft: from FORE, counter-clockwise seen from above (PORT 90°). Or exhaust / manoeuvring side?
4. **Measuring positions.** UMA's standard measuring depths per engine type: the VAK positions for the table.
5. **Engines.** Turning-gear speed (0.1 rpm assumed), crown-to-liner-top and crown-to-valve clearance at TDC, port count and phase relative to FORE, for the types UMA services.
6. **Hardware risks to accept or resolve.** Battery rating at 80 °C, laser class 3B with interlock, and testing BLE through the starting-air valve pocket.
7. **Height-sensor addition.** I added a small upward laser distance module (DECISIONS E9) because multizone ToF alone gave 7–15 mm height errors. OK to keep?

## Next three steps

1. **Bench prototype of the L optical module on the rail** (camera, lens, laser, Invar bench, sapphire windows). Measure real point noise, distortion and thermal drift in a chamber, then feed them into `CameraTruth` and re-validate. This is the biggest remaining uncertainty.
2. **Mechanical mock-up at 350 × 100 × 20 mm, plus the electronics on the chosen SoM and nRF52840.** Run `ScanController` on hardware with a linear stage standing in for the piston and a 600 mm ring as the liner.
3. **One trial cylinder against the VAK / LDM**, with Brian's decisions on questions 1–4 applied to the report.

## Budget

I estimate roughly USD {SPEND} of credit used, from tool-call and context volume. The session does not show exact spend. Work stopped well before the USD 85 line.
