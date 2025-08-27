"""Manifold sampling utilities.

This module defines a metaclass :class:`ManifoldMeta` that automatically
registers concrete manifold sampler classes so users can dynamically
instantiate them by name. Each manifold class implements a ``sample``
method returning an ``(n, ambient_dim)`` NumPy array.

The implementations are Python translations of the provided R sampling
functions (r*). Two variants often exist – a *core* version that returns
the geometric coordinates only, and an *embedded* version that augments
coordinates with the pattern ``[sin(X) | X | X**2]`` used in the original
R code. Embedded variants keep the original R function name semantics:

	rtorus  -> Torus (embedded)
	rtorus2 -> TorusCore (core)

Usage examples
--------------
>>> from manifolds.manifold import get_manifold
>>> sphere = get_manifold('sphere', dim=5)  # Sphere embedded variant
>>> X = sphere.sample(100, random_state=0)
>>> X.shape  # (100, 15) because embedded: 3 * base_dim

>>> torus = get_manifold('toruscore')  # core torus in R: rtorus2
>>> torus.sample(10).shape  # (10, 3)

Factory helpers
---------------
``get_manifold(name, **kwargs)`` returns an instance.
``list_manifolds()`` enumerates available registry keys.

All samplers accept an optional ``random_state`` (int | np.random.Generator)
argument for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union
import math
import numpy as np

RandomStateLike = Union[int, np.random.Generator, None]


def _rng(random_state: RandomStateLike = None) -> np.random.Generator:
	if isinstance(random_state, np.random.Generator):
		return random_state
	if random_state is None:
		return np.random.default_rng()
	return np.random.default_rng(random_state)


def _embed_trig_poly(X: np.ndarray) -> np.ndarray:
	"""Return concatenation [sin(X) | X | X**2]."""
	return np.hstack([np.sin(X), X, X ** 2])


class ManifoldMeta(type):
	"""Metaclass that auto-registers concrete manifold classes.

	A class is registered if it defines ``abstract = False`` (default) and
	implements a ``sample`` method.
	"""

	registry: Dict[str, Type['BaseManifold']] = {}

	def __new__(mcls, name, bases, namespace, **kwargs):  # type: ignore[override]
		cls = super().__new__(mcls, name, bases, dict(namespace))
		abstract = namespace.get('abstract', False)
		if not abstract and name != 'BaseManifold':
			key = getattr(cls, 'name', name).lower()
			ManifoldMeta.registry[key] = cls  # register
		return cls

	@classmethod
	def get(cls, key: str) -> Type['BaseManifold']:
		try:
			return cls.registry[key.lower()]
		except KeyError as exc:
			raise KeyError(f"Unknown manifold '{key}'. Available: {sorted(cls.registry)}") from exc


class BaseManifold(metaclass=ManifoldMeta):
	"""Abstract base manifold sampler.

	Subclasses define:
	- ``base_dim``: intrinsic/base dimension (input dimension before embedding)
	- ``embedded``: bool, whether trig-poly embedding is applied
	- ``sample(n, random_state=None)``: returns ndarray shape (n, ambient_dim)
	"""

	abstract = True
	name: str
	base_dim: int
	embedded: bool = False

	def ambient_dim(self) -> int:
		return (3 * self.base_dim) if self.embedded else self.base_dim

	# Allow repr to show dims
	def __repr__(self) -> str:  # pragma: no cover - simple repr
		kind = 'embedded' if self.embedded else 'core'
		return f"<{self.__class__.__name__} base_dim={self.base_dim} ambient_dim={self.ambient_dim()} {kind}>"

	# To be implemented by subclasses
	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:  # pragma: no cover - abstract
		raise NotImplementedError


# ----------------------- Helper / internal distributions -------------------- #


class MBurr(BaseManifold):
	"""Multivariate Burr-like distribution used by some paraboloid manifolds.

	Implements the R function rmburr(n, d, alfa).
	Returns shape (n, d).
	"""

	abstract = False
	name = 'mburr'

	def __init__(self, dim: int, alpha: float = 1.0):
		if dim < 1:
			raise ValueError('dim must be >= 1')
		self.base_dim = dim
		self.alpha = float(alpha)
		self.embedded = False

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		rng = _rng(random_state)
		x = rng.exponential(1.0, size=(n, self.base_dim))
		y = rng.gamma(shape=self.alpha, scale=1.0, size=n)  # (n,)
		x = x / y[:, None]
		x = (1 + x) ** (-self.alpha)
		return x


# ----------------------------- Paraboloid family --------------------------- #


class Paraboloid(BaseManifold):
	"""Embedded paraboloid manifold (R: rpara).

	Underlying construction: sample (d-1)-dim MBurr (alpha=1), compute
	z = sum(x^2) added as final coordinate -> base_dim = d
	Embedded output dimension = 3 * d.
	"""

	name = 'para'

	def __init__(self, dim: int):
		if dim < 2:
			raise ValueError('dim must be >= 2 (because underlying uses dim-1)')
		self.base_dim = dim
		self.embedded = True

	def _core(self, n: int, rng: np.random.Generator) -> np.ndarray:
		x = MBurr(self.base_dim - 1, alpha=1.0).sample(n, rng)
		z = np.sum(x ** 2, axis=1, keepdims=True)
		y = np.hstack([x, z])  # (n, base_dim)
		return y

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		rng = _rng(random_state)
		y = self._core(n, rng)
		return _embed_trig_poly(y)


class ParaboloidCore(Paraboloid):
	"""Core (non-embedded) paraboloid (R: rpara2)."""

	name = 'paracore'

	def __init__(self, dim: int):
		super().__init__(dim)
		self.embedded = False

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		return self._core(n, _rng(random_state))


class ParaboloidUniformCore(BaseManifold):
	"""Core paraboloid with uniform base (R: rparaunif)."""

	name = 'paraunif'

	def __init__(self, dim: int):
		if dim < 2:
			raise ValueError('dim must be >= 2')
		self.base_dim = dim
		self.embedded = False

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		rng = _rng(random_state)
		x = rng.uniform(0, 1, size=(n, self.base_dim - 1))
		z = np.sum(x ** 2, axis=1, keepdims=True)
		y = np.hstack([x, z])
		return y


# ----------------------------- Swiss roll family --------------------------- #


class SwissRoll(BaseManifold):
	"""Embedded swiss roll variant (R: rswiss)."""

	name = 'swiss'

	def __init__(self):
		self.base_dim = 3  # after constructing z matrix
		self.embedded = True

	def _core(self, n: int, rng: np.random.Generator) -> np.ndarray:
		nc = n // 4
		x = rng.normal(size=(n, 2))
		# quadrant translations replicate R logic
		idx1 = slice(0, nc)
		idx2 = slice(nc, 2 * nc)
		idx3 = slice(2 * nc, 3 * nc)
		idx4 = slice(3 * nc, n)
		x[idx1] += np.array([7.5, 7.5])
		x[idx2] += np.array([7.5, 12.5])
		x[idx3] += np.array([12.5, 7.5])
		x[idx4] += np.array([12.5, 12.5])
		tmp = x[:, 0]
		z1 = tmp * np.cos(tmp)
		z2 = x[:, 1]
		z3 = tmp * np.sin(tmp)
		z = np.vstack([z1, z2, z3]).T  # (n,3)
		return z

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		z = self._core(n, _rng(random_state))
		return _embed_trig_poly(z)


class SwissRollCore(SwissRoll):
	name = 'swisscore'

	def __init__(self):
		super().__init__()
		self.embedded = False

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		return self._core(n, _rng(random_state))


# ------------------------------- Sphere family ----------------------------- #


class Sphere(BaseManifold):
	"""Points on S^{d-1} then embedded (R: rsphere)."""

	name = 'sphere'

	def __init__(self, dim: int):
		if dim < 2:
			raise ValueError('dim must be >= 2')
		self.base_dim = dim
		self.embedded = True

	def _core(self, n: int, rng: np.random.Generator) -> np.ndarray:
		x = rng.normal(size=(n, self.base_dim))
		norms = np.linalg.norm(x, axis=1, keepdims=True)
		x /= norms
		return x

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		return _embed_trig_poly(self._core(n, _rng(random_state)))


class SphereCore(Sphere):
	name = 'spherecore'

	def __init__(self, dim: int):
		super().__init__(dim)
		self.embedded = False

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		return self._core(n, _rng(random_state))


# -------------------------------- Torus family ----------------------------- #


class Torus(BaseManifold):
	"""Embedded torus in R^3 (R: rtorus). Parameters: major radius R0=2, minor radius r=1."""

	name = 'torus'

	def __init__(self):
		self.base_dim = 3
		self.embedded = True

	def _core(self, n: int, rng: np.random.Generator) -> np.ndarray:
		x = rng.uniform(0, 2 * math.pi, size=n)
		y = rng.uniform(0, 2 * math.pi, size=n)
		X = np.zeros((n, 3))
		X[:, 0] = (2 + np.cos(y)) * np.cos(x)
		X[:, 1] = (2 + np.cos(y)) * np.sin(x)
		X[:, 2] = np.sin(y)
		return X

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		return _embed_trig_poly(self._core(n, _rng(random_state)))


class TorusCore(Torus):
	name = 'toruscore'

	def __init__(self):
		super().__init__()
		self.embedded = False

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		return self._core(n, _rng(random_state))


# ------------------------------- Variety family ---------------------------- #


class Variety(BaseManifold):
	"""Embedded variety (R: rvariedad)."""

	name = 'variety'

	def __init__(self):
		self.base_dim = 3
		self.embedded = True

	def _core(self, n: int, rng: np.random.Generator) -> np.ndarray:
		x = rng.uniform(-5, 5, size=n)
		y = rng.normal(loc=0, scale=1, size=n)
		Z = np.zeros((n, 3))
		Z[:, 0] = x
		Z[:, 1] = y * np.cos(x)
		Z[:, 2] = x * np.sin(x)
		return Z

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		Z = self._core(n, _rng(random_state))
		return np.hstack([Z, np.sin(Z)])  # matches R: cbind(z, sin(z))


class VarietyCore(Variety):
	name = 'varietycore'

	def __init__(self):
		super().__init__()
		self.embedded = False

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		return self._core(n, _rng(random_state))


# ------------------------------- Uniform cube ------------------------------ #


class UniformCube(BaseManifold):
	"""Uniform in [0,1]^d (R: rmunif)."""

	name = 'uniformcube'

	def __init__(self, dim: int):
		if dim < 1:
			raise ValueError('dim must be >= 1')
		self.base_dim = dim
		self.embedded = False

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		return _rng(random_state).uniform(0, 1, size=(n, self.base_dim))


# ------------------------------- Möbius strip ------------------------------ #


class Mobius(BaseManifold):
	"""Embedded Möbius strip (R: rmobius)."""

	name = 'mobius'

	def __init__(self):
		self.base_dim = 3
		self.embedded = True

	def _core(self, n: int, rng: np.random.Generator) -> np.ndarray:
		U = rng.uniform(-0.5, 0.5, size=n)
		V = rng.uniform(0, 2 * math.pi, size=n)
		x1 = (1 + U * np.cos(V * 5)) * np.cos(V)
		x2 = (1 + U * np.cos(V * 5)) * np.sin(V)
		x3 = U * np.sin(V * 5)
		return np.vstack([x1, x2, x3]).T

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		X = self._core(n, _rng(random_state))
		return _embed_trig_poly(X)


class MobiusCore(Mobius):
	name = 'mobiuscore'

	def __init__(self):
		super().__init__()
		self.embedded = False

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		return self._core(n, _rng(random_state))


# ------------------------- Gaussian mixture & normal ----------------------- #


class GaussianMixture(BaseManifold):
	"""Mixture of 3 spherical Gaussians at -10, 0, 10 (R: rmixture).

	Weights: 0.3, 0.5, 0.2.
	"""

	name = 'mixture'

	def __init__(self, dim: int, means: Optional[List[float]] = None, weights: Optional[List[float]] = None, var: float = 0.5):
		if dim < 1:
			raise ValueError('dim must be >= 1')
		self.base_dim = dim
		self.embedded = False
		self.means = means if means is not None else [0.0, 10.0, -10.0]
		self.weights = np.array(weights if weights is not None else [0.3, 0.5, 0.2], dtype=float)
		if not np.isclose(self.weights.sum(), 1.0):
			raise ValueError('weights must sum to 1')
		self.var = float(var)

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		rng = _rng(random_state)
		comps = rng.choice(len(self.means), size=n, p=self.weights)
		X = np.empty((n, self.base_dim))
		for i, mean_idx in enumerate(comps):
			mean = np.full(self.base_dim, self.means[mean_idx])
			X[i] = rng.normal(loc=mean, scale=math.sqrt(self.var))
		return X


class GaussianNormal(BaseManifold):
	"""Standard multivariate normal N(0, I) (R: rmnorm)."""

	name = 'mnormal'

	def __init__(self, dim: int):
		if dim < 1:
			raise ValueError('dim must be >= 1')
		self.base_dim = dim
		self.embedded = False

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		return _rng(random_state).normal(size=(n, self.base_dim))


# ----------------------------- Uniform unit ball --------------------------- #


class UniformBall(BaseManifold):
	"""Uniform in unit ball in R^d (R: runiball)."""

	name = 'uniball'

	def __init__(self, dim: int):
		if dim < 1:
			raise ValueError('dim must be >= 1')
		self.base_dim = dim
		self.embedded = False

	def sample(self, n: int, random_state: RandomStateLike = None) -> np.ndarray:
		rng = _rng(random_state)
		X = rng.normal(size=(n, self.base_dim))
		X /= np.linalg.norm(X, axis=1, keepdims=True)
		r = rng.uniform(size=n) ** (1 / self.base_dim)
		X *= r[:, None]
		return X


# ------------------------------ Factory helpers ---------------------------- #


def get_manifold(name: str, **kwargs) -> BaseManifold:
	"""Instantiate a registered manifold by name.

	Parameters
	----------
	name : str
		Registry key (case-insensitive). See ``list_manifolds``.
	**kwargs : dict
		Passed to the manifold constructor (e.g., dim=, alpha=, ...).
	"""

	cls = ManifoldMeta.get(name)
	return cls(**kwargs)  # type: ignore[arg-type]


def list_manifolds() -> List[str]:
	return sorted(ManifoldMeta.registry.keys())


__all__ = [
	'BaseManifold', 'ManifoldMeta', 'get_manifold', 'list_manifolds',
	# classes
	'MBurr', 'Paraboloid', 'ParaboloidCore', 'ParaboloidUniformCore',
	'SwissRoll', 'SwissRollCore', 'Sphere', 'SphereCore', 'Torus', 'TorusCore',
	'Variety', 'VarietyCore', 'UniformCube', 'Mobius', 'MobiusCore',
	'GaussianMixture', 'GaussianNormal', 'UniformBall'
]

