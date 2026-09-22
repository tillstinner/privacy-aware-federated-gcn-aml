from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TenSEALContextConfig:
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: tuple[int, ...] = (60, 40, 40, 60)
    global_scale: float = 2**40


@dataclass(frozen=True)
class TenSEALContextBundle:
    """Local-simulation CKKS contexts with explicit server/client separation."""

    config: TenSEALContextConfig
    public_context: Any
    secret_context: Any
    public_serialized_bytes: int
    secret_serialized_bytes: int


def _import_tenseal():
    try:
        import tenseal as ts
    except ImportError as exc:
        raise ImportError(
            "TenSEAL is required for --feature-pretrain-transport he. "
            "Install tenseal in the active environment or use --feature-pretrain-transport plain."
        ) from exc
    return ts


def create_tenseal_context_bundle(config: TenSEALContextConfig | None = None) -> TenSEALContextBundle:
    ts = _import_tenseal()
    config = config or TenSEALContextConfig()
    secret_context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=config.poly_modulus_degree,
        coeff_mod_bit_sizes=list(config.coeff_mod_bit_sizes),
    )
    secret_context.global_scale = config.global_scale
    public_bytes = secret_context.serialize(save_secret_key=False)
    secret_bytes = secret_context.serialize(save_secret_key=True)
    public_context = ts.context_from(public_bytes)
    client_context = ts.context_from(secret_bytes)
    return TenSEALContextBundle(
        config=config,
        public_context=public_context,
        secret_context=client_context,
        public_serialized_bytes=len(public_bytes),
        secret_serialized_bytes=len(secret_bytes),
    )


def ckks_vector(context, values: list[float]):
    ts = _import_tenseal()
    return ts.ckks_vector(context, values)


def ckks_vector_from(context, payload: bytes):
    ts = _import_tenseal()
    return ts.ckks_vector_from(context, payload)
