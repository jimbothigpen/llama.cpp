#!/usr/bin/env python3
# TODO 204 — InnerQ x TCQ decode correctness: synthetic numerical proof.
#
# Models the EXACT fork rotation used by the TCQ encode and the Q-side
# compensation (turbo-quant.cuh turbo_rotate_forward / turbo-wht.cu):
#     rotate_forward(x) = SIGNS2 (.) FWHT( SIGNS1 (.) x )            (orthonormal)
#     rotate_inverse(y) = SIGNS1 (.) FWHT( SIGNS2 (.) y )            (its inverse)
# FWHT here is the normalized (orthonormal) Walsh-Hadamard transform: H/sqrt(n),
# H symmetric and H H = n I, so the normalized transform is its own inverse.
#
# Encode (set-rows.cu, InnerQ active): stored = trellis( rotate_forward( scale (.) K ) )
#   -> decode produces approx rotate_forward( scale (.) K )   [ "R" below ]
#
# Two ways to recover dot(Q,K) from R:
#   A) Q-SIDE COMPENSATION (what the fork ships, graph line ~2546):
#        Qrot = rotate_forward( scale_inv (.) Q );   score = dot(Qrot, R)
#   B) BLOCK-LEVEL DECODE (what TODO 204 proposes):
#        Korig = scale_inv (.) rotate_inverse(R);   then re-rotate Krot = rotate_forward(Korig)
#        Qrot_plain = rotate_forward(Q);   score = dot(Qrot_plain, Krot)
#
# Claim under test: with injected NON-UNIT per-channel scale, BOTH A and B recover
# dot(Q,K) to fp tolerance; therefore B adds no correctness over the already-shipped A.
# Also demonstrates that the "per-bin scalar inverse inside the dequant" is unsound.

import numpy as np

np.random.seed(204)
N = 128  # TCQ block / FWHT group size in the fork

def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H

H = hadamard(N) / np.sqrt(N)          # normalized -> orthonormal, H@H = I
assert np.allclose(H @ H, np.eye(N), atol=1e-6)

s1 = np.random.choice([-1.0, 1.0], size=N)   # stand-ins for TURBO_WHT_SIGNS1/2
s2 = np.random.choice([-1.0, 1.0], size=N)

def rotate_forward(x):
    return s2 * (H @ (s1 * x))

def rotate_inverse(y):
    return s1 * (H @ (s2 * y))

# sanity: inverse really inverts, and rotation is orthonormal (dot-preserving)
xt = np.random.randn(N)
assert np.allclose(rotate_inverse(rotate_forward(xt)), xt, atol=1e-5)
a, b = np.random.randn(N), np.random.randn(N)
assert np.allclose(np.dot(rotate_forward(a), rotate_forward(b)), np.dot(a, b), atol=1e-5)

def run_trial(quantize=False, qbits=3):
    Q = np.random.randn(N).astype(np.float64)
    K = np.random.randn(N).astype(np.float64)
    # INJECTED non-unit per-channel scale (InnerQ equalization), strongly non-unit.
    scale = np.exp(np.random.uniform(-1.0, 1.0, size=N))   # ~0.37 .. 2.7
    scale_inv = 1.0 / scale

    ref = np.dot(Q, K)

    # ---- encode: stored rotated, optionally TCQ-quantized in the rotated domain ----
    R = rotate_forward(scale * K)
    if quantize:
        # crude uniform quant of the rotated bins to emulate TCQ codebook error
        step = (R.max() - R.min()) / (2 ** qbits - 1) + 1e-12
        R = np.round(R / step) * step

    # ---- A) Q-side compensation (shipped) ----
    Qrot_A = rotate_forward(scale_inv * Q)
    score_A = np.dot(Qrot_A, R)

    # ---- B) block-level decode (TODO 204 proposal) ----
    Korig = scale_inv * rotate_inverse(R)        # back to original basis, scale removed
    Krot_B = rotate_forward(Korig)               # "re-rotate as needed"
    Qrot_plain = rotate_forward(Q)
    score_B = np.dot(Qrot_plain, Krot_B)
    score_B_direct = np.dot(Q, Korig)            # equivalent: dot in original basis

    # ---- C) UNSOUND per-bin inverse inside the dequant (what §-FLAG-A rejected) ----
    # Try to "fix" each rotated bin t by a single scalar derived from scale. No such
    # per-bin scalar exists because a diagonal does not commute with FWHT. We try the
    # most charitable per-bin factor (rotate_forward of scale_inv) applied elementwise:
    perbin_factor = rotate_forward(scale_inv) / (rotate_forward(np.ones(N)) + 1e-12)
    R_perbin = R * perbin_factor
    score_perbin = np.dot(rotate_forward(Q), R_perbin)

    return ref, score_A, score_B, score_B_direct, score_perbin

print("=== EXACT (no quantization) — pure inverse-math check ===")
maxerr_A = maxerr_B = maxerr_pb = 0.0
for _ in range(2000):
    ref, sA, sB, sBd, spb = run_trial(quantize=False)
    maxerr_A  = max(maxerr_A,  abs(sA  - ref) / (abs(ref) + 1e-9))
    maxerr_B  = max(maxerr_B,  abs(sB  - ref) / (abs(ref) + 1e-9))
    maxerr_pb = max(maxerr_pb, abs(spb - ref) / (abs(ref) + 1e-9))
print(f"  A) Q-side compensation   max rel err vs dot(Q,K): {maxerr_A:.3e}   -> {'RECOVERS' if maxerr_A < 1e-5 else 'FAILS'}")
print(f"  B) block-level decode    max rel err vs dot(Q,K): {maxerr_B:.3e}   -> {'RECOVERS' if maxerr_B < 1e-5 else 'FAILS'}")
print(f"  C) per-bin scalar inverse max rel err vs dot(Q,K): {maxerr_pb:.3e}  -> {'RECOVERS' if maxerr_pb < 1e-5 else 'UNSOUND (as expected)'}")

print("\n=== WITH 3-bit quantization in the rotated domain (TCQ-like) ===")
errs_A, errs_B = [], []
for _ in range(2000):
    ref, sA, sB, sBd, spb = run_trial(quantize=True, qbits=3)
    errs_A.append(abs(sA - ref))
    errs_B.append(abs(sB - ref))
errs_A = np.array(errs_A); errs_B = np.array(errs_B)
print(f"  A) Q-side comp   mean|err|={errs_A.mean():.4f}  rms={np.sqrt((errs_A**2).mean()):.4f}")
print(f"  B) block-decode  mean|err|={errs_B.mean():.4f}  rms={np.sqrt((errs_B**2).mean()):.4f}")
print(f"  ratio B/A rms = {np.sqrt((errs_B**2).mean())/ (np.sqrt((errs_A**2).mean())+1e-12):.4f}  (≈1 -> no quantization-accuracy advantage)")

print("\nCONCLUSION: A (shipped Q-side compensation) and B (proposed block decode) both")
print("recover dot(Q,K) to fp tolerance and carry statistically identical quantization")
print("error. The block decode is mathematically redundant with the shipped correction.")
