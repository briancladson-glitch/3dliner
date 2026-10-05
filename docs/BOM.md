# Bill of materials (final, prototype quantity 1)

Prices are rough **USD estimates for single units** in 2026. They are for budgeting only and have not been quoted. Part names are *examples of a class*: check each datasheet before ordering. Machine-readable version: `docs/bom.csv`.

## Totals

| Section | USD |
|---|---:|
| A Optical modules (x3: S/M/L) | 1,596 |
| B Rotating head | 625 |
| C Electronics | 643 |
| D Mechanics | 1,030 |
| E Base station | 1,415 |
| F Calibration (one-off, lab) | 24,000 |
| **Scanner (A–D), parts only** | **3,894** |
| **Base station (E)** | **1,415** |
| **Calibration equipment (F), one-off** | **24,000** |

Not included: engineering and prototyping labour, PCB NRE, certification (laser class, ATEX if UMA needs it), and spares. Expect two prototype iterations.

## Items

### A Optical modules (x3: S/M/L)

| Item | Example / class | Qty | Unit | Total | Notes |
|---|---|---:|---:|---:|---|
| Global-shutter mono camera module, 1280x800, 3 µm, MIPI CSI-2, ROI ≥200 fps | OV9281-class board-level module on flex | 3 | 45 | 135 | QE at 405 nm and ROI frame rate to confirm |
| M12 lens, low distortion, fixed focus, glued | f 6 mm (S), 8 mm (M), 12 mm (L), F2.8 | 3 | 35 | 105 | one per module |
| 405 nm band-pass filter, 10 nm FWHM | Ø8–12 mm interference filter | 3 | 40 | 120 | rejects ambient / glow |
| 405 nm line laser module with Powell lens, 20 mW, TTL | fan 15° / 13° / 10° per preset | 3 | 90 | 270 | laser class 3R/3B → interlock E7 |
| Invar 36 optical bench + kinematic mount (3 balls / vee) | CNC, stress-relieved | 3 | 250 | 750 | keeps baseline drift < 2 µm over 60 K |
| Sapphire windows, AR 405 nm (camera + laser) | Ø10 × 1 mm | 6 | 30 | 180 | scratch-proof against carbon/oil |
| Module ID EEPROM + 2 × TMP117 temperature sensors | I²C | 3 | 12 | 36 | calibration ID and thermal compensation inputs |

### B Rotating head

| Item | Example / class | Qty | Unit | Total | Notes |
|---|---|---:|---:|---:|---|
| Thin-section bearing (or crossed-roller), ~60 mm bore | Kaydon/Silverthin-class, 5 mm section | 1 | 120 | 120 | runout ≤ 2 µm |
| Flat gimbal BLDC motor, ≤ 10 mm tall | 40 mm class | 1 | 60 | 60 |  |
| FOC motor driver | DRV8316 / TMC6300 class | 1 | 15 | 15 |  |
| Absolute off-axis ring encoder, 18-bit | RLS AksIM-2 class | 1 | 350 | 350 | ≤ ±0.02° after self-calibration (to confirm) |
| Flex cable loop for ±190° (no slip ring) | custom flex | 1 | 80 | 80 |  |

### C Electronics

| Item | Example / class | Qty | Unit | Total | Notes |
|---|---|---:|---:|---:|---|
| Main compute module: MIPI CSI, NEON, 4 GB RAM, 32 GB eMMC | Raspberry Pi CM5 class | 1 | 90 | 90 | on-board peak detection + storage |
| Carrier PCB, 4-layer, assembled | custom, fits 100 mm width | 1 | 250 | 250 |  |
| Safety MCU + BLE 5 (coded PHY) | nRF52840 module | 1 | 15 | 15 | runs ToF monitor + buzzer independent of main SoM |
| Upward multizone ToF sensor 8×8 | ST VL53L8CX | 2 | 15 | 30 | secondary crush safety + height reference |
| Piezo buzzer ≥ 95 dB, 85 °C rated |  | 1 | 8 | 8 | audible at the scavenge door |
| 6-axis IMU | LSM6DSO / ICM-42688 class | 1 | 8 | 8 | handling detection, tilt cross-check |
| Upward narrow-beam laser distance module, ±1 mm, 10 Hz (height reference) | JRT/TF-class module ≤ 13 mm thick | 1 | 45 | 45 | device only moves vertically → beam always hits the same cover spot |
| Temperature sensors (body, battery) | TMP117 | 2 | 6 | 12 |  |
| Battery: high-temperature Li-ion pouch, 2S ~3000 mAh, discharge rated ≥ 85 °C |  | 1 | 120 | 120 | KEY RISK: temperature rating |
| BMS + fuel gauge + charger + sealed USB-C |  | 1 | 40 | 40 | data offload and charging |
| DC-DC converters, laser driver, protection |  | 1 | 25 | 25 |  |

### D Mechanics

| Item | Example / class | Qty | Unit | Total | Notes |
|---|---|---:|---:|---:|---|
| Body and arms 350 × 100 × 20 mm, arms ≤ 8 mm tall | Ti Gr5 or 17-4PH, CNC | 1 | 900 | 900 |  |
| High-temperature NdFeB magnet feet | N42SH (150 °C) | 6 | 5 | 30 | stop sliding / rotation on the crown |
| Aerogel insulation for battery and electronics |  | 1 | 40 | 40 |  |
| FKM O-rings, seals, fasteners |  | 1 | 40 | 40 |  |
| Retrieval lanyard and handle |  | 1 | 20 | 20 |  |

### E Base station

| Item | Example / class | Qty | Unit | Total | Notes |
|---|---|---:|---:|---:|---|
| Rugged tablet or laptop |  | 1 | 900 | 900 | runs processing + report |
| BLE dongle with u.FL | nRF52840 dongle | 1 | 15 | 15 |  |
| Antenna probe for the starting-air valve pocket + 5 m low-loss coax | stainless tube, monopole | 1 | 250 | 250 | fit to UMA's valve pockets |
| Transport case with foam + charging dock | Peli-type | 1 | 250 | 250 |  |

### F Calibration (one-off, lab)

| Item | Example / class | Qty | Unit | Total | Notes |
|---|---|---:|---:|---:|---|
| Precision linear stage, 600 mm travel, linear scale ≤ 5 µm |  | 1 | 6,000 | 6,000 | rail for triangulation calibration |
| Flat matte ceramic target, flat to 5 µm | 150 × 150 mm | 1 | 600 | 600 |  |
| Master rings Ø480 / Ø660 / Ø880, steel, CMM-certified ±2 µm | 60 mm high | 3 | 5,000 | 15,000 | R0/h calibration + field ring checks |
| Climatic chamber campaign 20–80 °C (rental/test house) |  | 1 | 2,000 | 2,000 | thermal coefficients |
| Reference thermometer PT100 ±0.05 K |  | 1 | 400 | 400 | ring temperature |

## Why these choices

- **Three optical modules** (S/M/L, docs/DECISIONS.md E1). Each is fixed-focus and calibrated on its own, and swaps on a kinematic mount. One body covers 400–980 mm.
- **Invar bench, sapphire windows, chamber-calibrated thermal model.** Accuracy has to hold with the liner at 80 °C. Uncompensated drift is about 1 mm (VALIDATION.md).
- **18-bit absolute ring encoder and thin-section bearing.** Angle and runout errors stay at the micron level even when the device sits 20–30 mm off centre.
- **Separate safety MCU (nRF52840)** for the ToF monitor and buzzer. Crush protection works even if the main computer or the radio fails. It also carries BLE to the base station.
- **Magnet feet.** The head's reaction torque must not turn the device on the crown, because yaw is referenced once.

## Key hardware risks

1. **Battery at 80 °C.** Most Li-ion cells are rated to 60 °C discharge. Options: high-temperature cells, plus aerogel insulation, plus keeping the soak short.
2. **20 mm height.** The M12 lens, sensor board and window must fit under 20 mm, with the optical axis about 13 mm above the crown. A mechanical mock-up should come first.
3. **Laser class.** 20 mW at 405 nm is class 3B. Keep the interlock (laser on only inside a cylinder), or check whether a lower power is enough on real liner surfaces.
4. **BLE through the starting-air valve pocket.** Test the radio link on a real cover early. The design does not need it for safety.
