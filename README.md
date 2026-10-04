# Encrypted State-Feedback Control Simulation

Software simulation of a state-feedback controller executing on homomorphically encrypted data.

## Overview

This project demonstrates encrypted state-feedback control using BGV (Brakerski-Gentry-Vaikuntanathan) homomorphic encryption.

The basic control law is:

    u = -Kx

where `x` is the system state and `K` is the state-feedback gain.

In the encrypted version, the state is encrypted before the controller computation:

    Enc(x) · (-K) = Enc(u)

**Plant: a 9-mass spring-damper chain (18 states)**, single actuator applied to the last
mass, single sensor measuring the last mass's position. This is the same architecture as
the original 2-state mass-spring-damper demo -- it's still a single-input, single-output
(SISO) controller with one gain row `K` and one HE dot product per control step -- just
with 9 masses daisy-chained together instead of 1, so the state vector grew from 2 to 18
without changing the control/encryption architecture itself.

    wall --[k,c]-- m1 --[k,c]-- m2 --[k,c]-- ... --[k,c]-- m9 <-- u(t)

The simulation includes:

- A 9-mass chain state-space plant model (18-state system)
- LQR state-feedback gain calculation (see "Why LQR?" below)
- Plaintext controller as a reference
- Encrypted controller using OpenFHE / BGV
- Correctness and timing analysis
- Comparison plots

## Prerequisites

- Python 3.7+
- `pip` or equivalent package manager

## Installation

Install the required packages:

```bash
pip install openfhe scipy matplotlib numpy
```

If `pip` is not recognized:

```bash
python -m pip install openfhe scipy matplotlib numpy
```

## Running

From the project root:

```bash
cd controller_sim
```

### Run the complete simulation

```bash
python compare_and_plot.py
```

or:

```bash
python3 compare_and_plot.py
```

This runs the plaintext and encrypted simulations, compares their results, and generates the plots and numerical summary in `results/`.

### Run only the encrypted controller

To directly see the BGV encrypted controller execution:

```bash
python encrypted_sim.py
```

or:

```bash
python3 encrypted_sim.py
```

This prints the execution of each control step, including:

- State being fixed-point encoded and encrypted
- BGV encryption
- Homomorphic computation `Enc(x) · (-K)`
- Encrypted control result
- Decryption
- Control value applied to the plant
- Encryption / HE / decryption timing

The plaintext `u = -Kx` calculation is also performed internally as a correctness check, but it is not used to generate the encrypted result or update the plant.

## Project structure

```
controller_sim/
├── state_space_model.py    # Plant model (9-mass chain) and controller gain
├── plaintext_sim.py        # Baseline (unencrypted) controller
├── encrypted_sim.py        # BGV encrypted controller
└── compare_and_plot.py     # Run both and generate plots

results/
├── state_response.png      # State trajectory comparison (first/middle/last mass)
├── control_signal.png      # Control input over time
├── error_vs_step.png       # BGV fixed-point quantization error (full 18-state norm)
├── timing_breakdown.png    # Operation timing
└── summary.txt             # Numerical results
```

## How it works

Each control step follows this flow:

```
Simulated state x   (18-dimensional)
    ↓
Fixed-point encode + BGV encryption
    ↓
Encrypted state Enc(x)
    ↓
Homomorphic computation
Enc(x) · (-K)
    ↓
Encrypted control Enc(u)   (still a single scalar u)
    ↓
Decrypt + rescale
    ↓
Control u
    ↓
Update plant
    ↓
Next state
```

The plaintext controller is used separately as a reference so that the decrypted HE result can be checked for correctness.

The actual control signal used to update the encrypted simulation comes from the decrypted homomorphic result.

## From 2 states to 18 states: what actually changed

Going from 2 states to 18 did **not** require changing the encryption logic. The BGV
dot product (packed multiply + `EvalSum` rotate-and-sum) works identically for a 2-element
or an 18-element vector. `encrypted_sim.py` and `plaintext_sim.py` are unchanged in their control logic.

What changed:

1. **`state_space_model.py`** -- `build_plant()` now builds an 18x18 `A` and 18x1 `B`
   for a chain of 9 masses (each mass = 2 states: position, velocity) instead of a
   single 2x2 mass-spring-damper. The chain is still driven by exactly one actuator
   (on the last mass) and measured at exactly one sensor (also the last mass), so the
   controller stays single-input, single-output.
2. **Gain design** -- switched from manual pole placement to **LQR**. See below.
3. **Logging / plots** -- the per-step trace and CSV columns now cover all 18 states
   instead of two hardcoded ones, and the state-response plot shows 3 representative
   masses (first, middle, last) along the chain instead of a single position/velocity
   pair, so the plot stays readable instead of overlaying 9 lines.

### Why LQR instead of manual pole placement?

The original 2-state demo picked two closed-loop poles by hand (`-4, -5`) and used
`scipy.signal.place_poles`. For an 18-state single-actuator chain this doesn't scale
well: some vibration modes of a long chain are only weakly reachable from a single
actuator at one end, so forcing all 18 poles to specific hand-picked locations demands
enormous feedback gains (we measured gains in the hundreds of millions), which then
overflows the fixed-point range of the BGV plaintext modulus. LQR (`scipy.linalg.solve_continuous_are`) instead
balances control effort against state error and produces a much smaller, better-scaled
gain matrix (`max|K| ≈ 57` here) while still stabilizing the whole chain -- it's the
same `u = -Kx` control law, just designed differently.

## Why BGV, and what it changes compared with BGV

BGV performs **exact modular integer arithmetic** (mod a plaintext modulus `t`), whereas BGV
performs approximate arithmetic on real numbers. The state and gain are real-valued, so they
are fixed-point encoded as integers around the encrypted computation:

    x_int = round(x * 2^15)          (encrypted by the sensor side)
    K_int = round(-K * 2^15)         (plaintext gain, controller side)
    u     = Dec( sum_i x_int[i] * K_int[i] ) / 2^30

The homomorphic part is exact. The only error versus the plaintext controller is the
fixed-point rounding of `x` and `K`.

Parameter notes:

- OpenFHE (BGV-RNS) is used because TenSEAL does not implement BGV.
- `t = 274878136321` (~2^38) is a prime with `t = 1 mod 2N`, which enables SIMD slot packing.
  The result must satisfy `|u| * 2^30 < t/2` (i.e. `|u| < 128`; the peak here is ~57). The
  simulation checks this every step and raises an error rather than silently wrapping mod `t`.
- Ring dimension is `N = 16384` (BGV version used 8192). With 128-bit security, N = 8192 only
  supports a plaintext modulus of about 22 bits, too small for this fixed-point range.
- Multiplicative depth 1 (one plaintext-ciphertext multiply).
- Precision can be traded against range by changing `SCALE_X`, `SCALE_K`, and
  `PLAINTEXT_MODULUS` at the top of `encrypted_sim.py`.

## Notes

- Each control step encrypts the current state independently.
- The simulation does not repeatedly operate on the same ciphertext across control steps, so the 200 control steps should not be interpreted as cumulative noise growth in a single ciphertext.
- BGV arithmetic is exact, so the difference between the decrypted result and the plaintext reference comes only from fixed-point rounding (measured here: max control error ~6.7e-3, max state error ~4.3e-4).
- The approximation error is measured directly from the simulation rather than assuming a fixed error range.
- The plant dynamics are simulated in plaintext; the state-feedback controller computation is the part performed homomorphically.
- Runtime depends on the BGV parameters. Larger ring dimensions and moduli increase computational cost.
- The number of masses is a single constant (`N_MASSES` in `state_space_model.py`) -- set it to 8 for a 16-state system, 9 for 18-state (default here), etc.

## Current scope

This is a software proof-of-concept for the encrypted controller.

The current implementation demonstrates:

- State-space modelling (9-mass chain, 18 states)
- State-feedback control (LQR design)
- BGV encryption (fixed-point encoded)
- Homomorphic controller computation
- Decryption of the control signal
- Correctness comparison
- Timing analysis

FPGA implementation and hardware-level optimizations such as NTT acceleration, pipelining, parallelism, and resource optimization are outside the scope of this simulation.
