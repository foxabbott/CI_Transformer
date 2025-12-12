from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, List, Any
import numpy as np

from .utils import rng, choose, sigmoid, safe_exp, softclip

Array = np.ndarray
MechanismFn = Callable[[Array, Array], Array]
# signature: f(parents_matrix, eps_vector) -> output_vector

@dataclass
class NoiseSpec:
    name: str
    params: Dict[str, Any]

    def sample(self, r: np.random.Generator, n: int) -> Array:
        p = self.params
        if self.name == "gaussian":
            return r.normal(loc=p.get("loc", 0.0), scale=p.get("scale", 1.0), size=n)
        if self.name == "laplace":
            return r.laplace(loc=p.get("loc", 0.0), scale=p.get("scale", 1.0), size=n)
        if self.name == "student_t":
            df = p.get("df", 3.0)
            scale = p.get("scale", 1.0)
            return r.standard_t(df=df, size=n) * scale
        if self.name == "uniform":
            a = p.get("low", -1.0); b = p.get("high", 1.0)
            return r.uniform(a, b, size=n)
        if self.name == "logistic":
            loc = p.get("loc", 0.0); scale = p.get("scale", 1.0)
            u = r.uniform(1e-6, 1-1e-6, size=n)
            return loc + scale * np.log(u/(1-u))
        if self.name == "mixture_gaussian":
            w = p.get("w", 0.5)
            m1, s1 = p.get("m1", -1.0), p.get("s1", 0.5)
            m2, s2 = p.get("m2", 1.0), p.get("s2", 0.5)
            m = r.random(n) < w
            out = np.empty(n)
            out[m] = r.normal(m1, s1, size=int(m.sum()))
            out[~m] = r.normal(m2, s2, size=int((~m).sum()))
            return out
        if self.name == "gamma_shifted":
            k = p.get("k", 2.0); theta = p.get("theta", 1.0); shift = p.get("shift", -1.0)
            return r.gamma(shape=k, scale=theta, size=n) + shift
        raise ValueError(f"Unknown noise: {self.name}")

def random_noise(r: np.random.Generator) -> NoiseSpec:
    choices = [
        ("gaussian", lambda: {"loc": 0.0, "scale": float(10**r.uniform(-0.3, 0.3))}),
        ("laplace", lambda: {"loc": 0.0, "scale": float(10**r.uniform(-0.3, 0.3))}),
        ("student_t", lambda: {"df": float(r.uniform(2.2, 10.0)), "scale": float(10**r.uniform(-0.3, 0.3))}),
        ("uniform", lambda: {"low": -float(r.uniform(0.5, 2.0)), "high": float(r.uniform(0.5, 2.0))}),
        ("logistic", lambda: {"loc": 0.0, "scale": float(10**r.uniform(-0.3, 0.3))}),
        ("mixture_gaussian", lambda: {"w": float(r.uniform(0.2, 0.8)),
                                      "m1": -float(r.uniform(0.5, 2.0)), "s1": float(r.uniform(0.2, 1.0)),
                                      "m2": float(r.uniform(0.5, 2.0)), "s2": float(r.uniform(0.2, 1.0))}),
        ("gamma_shifted", lambda: {"k": float(r.uniform(1.5, 5.0)), "theta": float(r.uniform(0.3, 1.5)), "shift": -float(r.uniform(0.0, 2.0))}),
    ]
    name, pfun = choose(r, choices)
    return NoiseSpec(name=name, params=pfun())

# ---- Mechanism templates ----
# Each mechanism returns a function f(P, eps) where P is (n, k) parent values, eps is (n,) noise.
# These are deliberately diverse: linear/nonlinear, additive/non-additive, heteroskedastic, etc.

def mech_linear(r: np.random.Generator, k: int) -> Tuple[MechanismFn, Dict[str, Any]]:
    w = r.normal(0, 1, size=k)
    b = float(r.normal(0, 0.5))
    scale_eps = float(10**r.uniform(-0.2, 0.4))
    def f(P: Array, eps: Array) -> Array:
        return P @ w + b + scale_eps * eps
    return f, {"type": "linear", "w": w.tolist(), "b": b, "scale_eps": scale_eps}

def mech_poly(r: np.random.Generator, k: int) -> Tuple[MechanismFn, Dict[str, Any]]:
    deg = int(r.integers(2, 5))
    W = r.normal(0, 1, size=(k, deg))
    b = float(r.normal(0, 0.5))
    scale_eps = float(10**r.uniform(-0.2, 0.4))
    def f(P: Array, eps: Array) -> Array:
        # sum_j sum_m W[j,m] * P_j^(m+1)
        out = np.zeros(P.shape[0])
        for m in range(deg):
            out += (P ** (m+1)) @ W[:, m]
        return softclip(out + b) + scale_eps * eps
    return f, {"type": "poly", "deg": deg, "W": W.tolist(), "b": b, "scale_eps": scale_eps}

def mech_trig(r: np.random.Generator, k: int) -> Tuple[MechanismFn, Dict[str, Any]]:
    w = r.normal(0, 1, size=k)
    a = float(r.uniform(0.5, 2.5))
    b = float(r.normal(0, 0.5))
    scale_eps = float(10**r.uniform(-0.2, 0.4))
    def f(P: Array, eps: Array) -> Array:
        t = P @ w
        return softclip(a * np.sin(t) + 0.3*np.cos(1.7*t) + b + scale_eps * eps)
    return f, {"type": "trig", "w": w.tolist(), "a": a, "b": b, "scale_eps": scale_eps}

def mech_rff_gp_like(r: np.random.Generator, k: int) -> Tuple[MechanismFn, Dict[str, Any]]:
    # Random Fourier Features approximating a stationary GP draw: f(x)=sum a_m cos(w_m^T x + b_m)
    m = int(r.integers(16, 64))
    W = r.normal(0, 1, size=(m, k)) / float(r.uniform(0.5, 2.0))
    phase = r.uniform(0, 2*np.pi, size=m)
    coeff = r.normal(0, 1, size=m) / np.sqrt(m)
    scale = float(r.uniform(0.5, 2.0))
    scale_eps = float(10**r.uniform(-0.2, 0.4))
    def f(P: Array, eps: Array) -> Array:
        proj = P @ W.T  # (n,m)
        feat = np.cos(proj + phase)
        out = scale * (feat @ coeff)
        return softclip(out) + scale_eps * eps
    return f, {"type": "rff_gp_like", "m": m, "scale": scale, "scale_eps": scale_eps}

def mech_mlp_random(r: np.random.Generator, k: int) -> Tuple[MechanismFn, Dict[str, Any]]:
    # Small random MLP; eps enters non-additively via gating.
    h = int(r.integers(8, 32))
    W1 = r.normal(0, 1, size=(k, h)) / np.sqrt(max(k,1))
    b1 = r.normal(0, 0.5, size=h)
    W2 = r.normal(0, 1, size=(h, 1)) / np.sqrt(h)
    b2 = float(r.normal(0, 0.5))
    gate = float(r.uniform(0.3, 1.5))
    scale_eps = float(10**r.uniform(-0.2, 0.4))
    def f(P: Array, eps: Array) -> Array:
        h1 = np.tanh(P @ W1 + b1)
        core = (h1 @ W2).reshape(-1) + b2
        g = sigmoid(gate * core)
        # eps enters by modulating the core (non-additive):
        return softclip(core + scale_eps * eps * (0.2 + g))
    return f, {"type": "mlp_random", "h": h, "gate": gate, "scale_eps": scale_eps}

def mech_heteroskedastic(r: np.random.Generator, k: int) -> Tuple[MechanismFn, Dict[str, Any]]:
    # Noise scale depends on parents; non-additive heteroskedasticity
    w = r.normal(0, 1, size=k)
    v = r.normal(0, 1, size=k)
    b = float(r.normal(0, 0.5))
    base = float(10**r.uniform(-0.2, 0.3))
    def f(P: Array, eps: Array) -> Array:
        mean = P @ w + b
        log_s = 0.3 * (P @ v)
        s = base * safe_exp(log_s, cap=4.0)
        return softclip(mean + s * eps)
    return f, {"type": "heteroskedastic", "w": w.tolist(), "v": v.tolist(), "b": b, "base": base}

def mech_multiplicative(r: np.random.Generator, k: int) -> Tuple[MechanismFn, Dict[str, Any]]:
    # Multiplicative noise: y = f(P) * (1 + s*eps)
    w = r.normal(0, 1, size=k)
    b = float(r.normal(0, 0.5))
    s = float(10**r.uniform(-0.5, 0.0))
    def f(P: Array, eps: Array) -> Array:
        core = np.tanh(P @ w + b)
        return softclip(core * (1.0 + s * eps))
    return f, {"type": "multiplicative", "w": w.tolist(), "b": b, "s": s}

def mech_ratio(r: np.random.Generator, k: int) -> Tuple[MechanismFn, Dict[str, Any]]:
    # Rational function with noise inside denominator
    w = r.normal(0, 1, size=k)
    v = r.normal(0, 1, size=k)
    b = float(r.normal(0, 0.5))
    c = float(r.uniform(0.5, 2.0))
    s = float(10**r.uniform(-0.2, 0.4))
    def f(P: Array, eps: Array) -> Array:
        num = P @ w + b
        den = c + np.abs(P @ v) + s*np.abs(eps)
        return softclip(num / den)
    return f, {"type": "ratio", "w": w.tolist(), "v": v.tolist(), "b": b, "c": c, "s": s}

MECHANISM_FACTORIES = [
    mech_linear,
    mech_poly,
    mech_trig,
    mech_rff_gp_like,
    mech_mlp_random,
    mech_heteroskedastic,
    mech_multiplicative,
    mech_ratio,
]

def random_mechanism(r: np.random.Generator, k: int) -> Tuple[MechanismFn, NoiseSpec, Dict[str, Any]]:
    """Pick a random mechanism and random noise distribution."""
    factory = choose(r, MECHANISM_FACTORIES)
    mech_fn, meta = factory(r, k=max(k, 1))
    noise = random_noise(r)
    return mech_fn, noise, meta
