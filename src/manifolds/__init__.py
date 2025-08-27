"""Manifold sampling package.

Convenience re-exports so users can do:
	from manifolds import get_manifold, list_manifolds
"""

from .manifold import (
	BaseManifold, ManifoldMeta, get_manifold, list_manifolds,
	MBurr, Paraboloid, ParaboloidCore, ParaboloidUniformCore,
	SwissRoll, SwissRollCore, Sphere, SphereCore, Torus, TorusCore,
	Variety, VarietyCore, UniformCube, Mobius, MobiusCore,
	GaussianMixture, GaussianNormal, UniformBall,
)

__all__ = [
	'BaseManifold', 'ManifoldMeta', 'get_manifold', 'list_manifolds',
	'MBurr', 'Paraboloid', 'ParaboloidCore', 'ParaboloidUniformCore',
	'SwissRoll', 'SwissRollCore', 'Sphere', 'SphereCore', 'Torus', 'TorusCore',
	'Variety', 'VarietyCore', 'UniformCube', 'Mobius', 'MobiusCore',
	'GaussianMixture', 'GaussianNormal', 'UniformBall'
]

