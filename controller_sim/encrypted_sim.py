"""
============================================================
ENCRYPTED CLOSED-LOOP SIMULATION
BGV homomorphic state-feedback controller
============================================================

This file is the actual execution/demo of the encrypted controller.

At every control step:

    1. The simulated plant has a real state x(k).
    2. The sensor side fixed-point encodes x(k) as integers and
       encrypts it using BGV (packed into one ciphertext).
    3. The controller evaluates Enc(x(k)) . (-K).
       The controller-side operation is performed on ciphertext.
    4. Only the resulting control signal is decrypted (and rescaled
       back from fixed-point to a real number).
    5. The plaintext plant is updated using that control signal.
    6. The process repeats.

The terminal output deliberately shows the execution pipeline so that
the simulation is observable instead of being only a final graph.

Important:
    This is a software proof-of-concept. The plant itself is simulated
    in plaintext. The state-feedback calculation is the part evaluated
    homomorphically. A real deployment would separate the sensor/
    encryption side, encrypted controller, and decryption/actuator side.

CKKS -> BGV: what changed
    * Library: TenSEAL (CKKS) -> OpenFHE (BGV, `pip install openfhe`).
      TenSEAL does not implement BGV.
    * BGV is an EXACT integer scheme (arithmetic mod plaintext modulus t),
      whereas CKKS is approximate over reals. So the real-valued state x
      and gain K are fixed-point encoded as integers before encryption:

            x_int = round(x * SCALE_X)        (encrypted)
            K_int = round(-K * SCALE_K)       (plaintext, controller side)
            u     = decrypt( sum_i x_int[i]*K_int[i] ) / (SCALE_X*SCALE_K)

      The HE arithmetic itself is exact; the only error is the
      fixed-point rounding of x and K.
    * t must be large enough that |u| * SCALE_X * SCALE_K < t/2, or the
      result wraps around modulo t. This is checked every step.
    * Dot product = packed plaintext multiply + EvalSum (rotate-and-sum),
      the same structure TenSEAL's `dot()` used internally.
============================================================
"""

import time
import numpy as np
from openfhe import (
    CCParamsBGVRNS,
    GenCryptoContext,
    PKESchemeFeature,
    SecurityLevel,
    BINARY,
    Serialize,
)

from state_space_model import build_plant, design_state_feedback, discretize, N_MASSES


# ------------------------------------------------------------
# BGV parameters
#   Plaintext modulus t must be a prime with t = 1 (mod 2N) so that
#   SIMD slot packing works. t = 274878136321 (~2^38) satisfies this
#   for N = 16384, and is the largest size OpenFHE accepts here.
#   Fixed-point scales: 2^15 each -> product scale 2^30, so values up
#   to |u| < t / (2 * 2^30) = 128 are representable (max |u| ~ 57).
# ------------------------------------------------------------
PLAINTEXT_MODULUS = 274878136321
RING_DIM = 16384
MULT_DEPTH = 1
SCALE_X = 2 ** 15
SCALE_K = 2 ** 15


class BGVContext:
    """Holds the OpenFHE crypto context, keys, and fixed-point scales."""

    def __init__(self, cc, keypair, scale_x, scale_k, plaintext_modulus, ring_dim):
        self.cc = cc
        self.public_key = keypair.publicKey
        self.secret_key = keypair.secretKey
        self.scale_x = scale_x
        self.scale_k = scale_k
        self.t = plaintext_modulus
        self.ring_dim = ring_dim


def make_context(
    plaintext_modulus=PLAINTEXT_MODULUS,
    ring_dim=RING_DIM,
    mult_depth=MULT_DEPTH,
    scale_x=SCALE_X,
    scale_k=SCALE_K,
):
    """Create the BGV context used by the simulation."""
    params = CCParamsBGVRNS()
    params.SetPlaintextModulus(plaintext_modulus)
    params.SetMultiplicativeDepth(mult_depth)
    params.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
    params.SetRingDim(ring_dim)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)   # needed for EvalSum

    keypair = cc.KeyGen()
    cc.EvalMultKeyGen(keypair.secretKey)      # relinearization key
    cc.EvalSumKeyGen(keypair.secretKey)       # rotation keys for rotate-and-sum

    return BGVContext(cc, keypair, scale_x, scale_k, plaintext_modulus, ring_dim)


def _fmt_vector(values, digits=6):
    values = np.asarray(values, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{v:.{digits}f}" for v in values) + "]"


def _ct_bytes(ct):
    try:
        return len(Serialize(ct, BINARY))
    except Exception:
        return -1


def simulate_encrypted(
    Ad,
    Bd,
    K,
    x0,
    steps,
    ctx=None,
    verbose=True,
    demo_steps=8,
    print_every=10,
):
    """
    Run the BGV encrypted-controller simulation.

    The optional terminal demonstration prints the first `demo_steps`
    control cycles and then every `print_every` steps. The complete
    numerical trajectory is returned to the caller.

    Returns:
        X         : plant state trajectory
        U         : control signal trajectory
        timings   : encryption/HE/decryption timing lists
        trace     : per-step execution metadata for CSV/reporting
    """
    if ctx is None:
        ctx = make_context()

    cc = ctx.cc
    n = Ad.shape[0]
    x = np.array(x0, dtype=float).reshape(n, 1)

    # Controller-side gain: fixed-point encoded -K, packed into one plaintext.
    # (The gain is not secret in this setup, so it stays unencrypted.)
    neg_K_int = [int(round(-v * ctx.scale_k)) for v in K.flatten()]
    neg_K_pt = cc.MakePackedPlaintext(neg_K_int)
    out_scale = float(ctx.scale_x * ctx.scale_k)
    t_half = ctx.t // 2

    X = np.zeros((steps + 1, n))
    U = np.zeros(steps)
    X[0] = x.flatten()

    timings = {"encrypt": [], "he_dot": [], "decrypt": []}
    trace = []

    if verbose:
        print("\n" + "=" * 72)
        print("BGV ENCRYPTED STATE-FEEDBACK EXECUTION")
        print("=" * 72)
        print(f"State dimension: {n}   Controller operation: Enc(u) = Enc(x) . (-K)")
        print(f"BGV: ring dim N={ctx.ring_dim}, plaintext modulus t={ctx.t}")
        print(f"Fixed-point scales: x * 2^{int(np.log2(ctx.scale_x))}, "
              f"K * 2^{int(np.log2(ctx.scale_k))}")
        print("The plaintext state is NOT used in the controller operation.")
        print("The terminal is an experiment observer; the simulated")
        print("sensor/plant values are shown here for demonstration.")
        print("-" * 72)

    for k in range(steps):
        time_s = k * 0.1

        # ------------------------------------------------------
        # SENSOR -> FIXED-POINT ENCODE -> ENCRYPTION
        # ------------------------------------------------------
        x_before = x.flatten().copy()

        t0 = time.perf_counter()
        x_int = [int(round(v * ctx.scale_x)) for v in x_before]
        x_pt = cc.MakePackedPlaintext(x_int)
        enc_x = cc.Encrypt(ctx.public_key, x_pt)
        t1 = time.perf_counter()

        # A serialized ciphertext size gives us concrete evidence that
        # a ciphertext object was created without printing secret data.
        enc_x_bytes = _ct_bytes(enc_x)

        # ------------------------------------------------------
        # ENCRYPTED CONTROLLER
        #   slot-wise multiply by packed (-K), then sum all slots
        #   (rotate-and-sum) so slot 0 holds sum_i x_i * (-K_i).
        # ------------------------------------------------------
        t2 = time.perf_counter()
        enc_prod = cc.EvalMult(enc_x, neg_K_pt)
        enc_u = cc.EvalSum(enc_prod, n)
        t3 = time.perf_counter()

        enc_u_bytes = _ct_bytes(enc_u)

        # ------------------------------------------------------
        # DECRYPT ONLY THE CONTROL ACTION
        # ------------------------------------------------------
        t3b = time.perf_counter()
        dec = cc.Decrypt(enc_u, ctx.secret_key)
        dec.SetLength(1)
        u_int = int(dec.GetPackedValue()[0])
        # Wrap-around guard: BGV works mod t, so a result outside
        # (-t/2, t/2) would silently be wrong. Fail loudly instead.
        if abs(u_int) >= t_half:
            raise OverflowError(
                f"step {k}: fixed-point result {u_int} exceeds t/2={t_half}; "
                "increase plaintext modulus or reduce scales."
            )
        u_val = u_int / out_scale
        t4 = time.perf_counter()

        encrypt_ms = (t1 - t0) * 1e3
        he_ms = (t3 - t2) * 1e3
        decrypt_ms = (t4 - t3b) * 1e3

        timings["encrypt"].append(encrypt_ms)
        timings["he_dot"].append(he_ms)
        timings["decrypt"].append(decrypt_ms)

        # Plaintext value is calculated ONLY for validation/measurement.
        # It is not used to produce u_val.
        u_plain_check = float((-K @ x)[0, 0])
        control_error = abs(u_plain_check - u_val)

        # ------------------------------------------------------
        # PHYSICAL PLANT UPDATE
        # ------------------------------------------------------
        x = Ad @ x + Bd * u_val
        X[k + 1] = x.flatten()
        U[k] = u_val

        state_change = float(np.linalg.norm(X[k + 1] - X[k]))

        trace.append(
            {
                "step": k,
                "time_s": time_s,
                "state_before": x_before.tolist(),   # full x(k), all n states
                "state_after": x.flatten().tolist(), # full x(k+1), all n states
                "u_encrypted": u_val,
                "u_plain_check": u_plain_check,
                "control_error": control_error,
                "state_change_norm": state_change,
                "encrypt_ms": encrypt_ms,
                "he_dot_ms": he_ms,
                "decrypt_ms": decrypt_ms,
                "enc_x_bytes": enc_x_bytes,
                "enc_u_bytes": enc_u_bytes,
            }
        )

        # ------------------------------------------------------
        # TERMINAL DEMONSTRATION
        # ------------------------------------------------------
        show = verbose and (k < demo_steps or (k + 1) % print_every == 0 or k == steps - 1)

        if show:
            print(f"\nSTEP {k:03d}   t = {time_s:5.2f} s")
            print(f"  Sensor state x(k)       = {_fmt_vector(x_before)}")
            print("  -> Fixed-point encoding + encrypting x(k) with BGV")
            print(f"  -> Ciphertext created    ({enc_x_bytes} bytes)" if enc_x_bytes >= 0
                  else "  -> Ciphertext created")
            print("  -> Controller: Enc(x) · (-K)")
            print(f"  -> Encrypted result      ({enc_u_bytes} bytes)" if enc_u_bytes >= 0
                  else "  -> Encrypted result created")
            print(f"  -> Decrypting control u  = {u_val:.9f}")
            print(f"  -> Plant update x(k+1)   = {_fmt_vector(x.flatten())}")
            print(
                f"  -> timing: encrypt={encrypt_ms:.3f} ms, "
                f"HE={he_ms:.3f} ms, decrypt={decrypt_ms:.3f} ms"
            )

    if verbose:
        avg_encrypt = np.mean(timings["encrypt"])
        avg_he = np.mean(timings["he_dot"])
        avg_decrypt = np.mean(timings["decrypt"])

        print("\n" + "=" * 72)
        print("EXECUTION COMPLETE")
        print("=" * 72)
        print(f"Control cycles completed : {steps}")
        print(f"Final state              : {_fmt_vector(X[-1])}")
        print(f"Maximum |u|              : {np.max(np.abs(U)):.9f}")
        print("\nAverage timing per control cycle:")
        print(f"  Encryption             : {avg_encrypt:.4f} ms")
        print(f"  Homomorphic dot product: {avg_he:.4f} ms")
        print(f"  Decryption             : {avg_decrypt:.4f} ms")
        print(
            f"  TOTAL                  : "
            f"{avg_encrypt + avg_he + avg_decrypt:.4f} ms"
        )

    return X, U, timings, trace


if __name__ == "__main__":
    A, B, C, D = build_plant()
    K = design_state_feedback(A, B)
    Ad, Bd, Cd, Dd = discretize(A, B, C, D, dt=0.1)

    n = A.shape[0]
    x0 = np.zeros(n)
    x0[2 * (N_MASSES - 1)] = 1.0

    print("=" * 72)
    print(f"{N_MASSES}-MASS CHAIN ({n} STATES) + BGV CONTROLLER")
    print("=" * 72)
    print(f"Initial state x(0) = {_fmt_vector(x0)}")
    print(f"State-feedback gain K = {_fmt_vector(K)}")
    print(f"Sampling time = 0.1 s")
    print("Building BGV context...")

    ctx = make_context()

    X, U, timings, trace = simulate_encrypted(
        Ad, Bd, K, x0, steps=200, ctx=ctx, verbose=True
    )

    print("\nThis run demonstrates:")
    print("  plaintext state -> BGV ciphertext -> HE control -> decrypted u")
    print("  -> plant update -> next state")
