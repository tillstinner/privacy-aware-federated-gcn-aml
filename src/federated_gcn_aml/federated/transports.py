from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from federated_gcn_aml.federated.encryption import (
    TenSEALContextBundle,
    TenSEALContextConfig,
    ckks_vector,
    ckks_vector_from,
    create_tenseal_context_bundle,
)
from federated_gcn_aml.federated.messages import BlockPlan, EncodedContributionBlock, EncodedPayloadBlock, FeatureContributionBlock


class FeaturePretrainTransport:
    name = "base"
    is_encrypted = False

    def encode_contribution_block(self, block: FeatureContributionBlock) -> EncodedContributionBlock:
        raise NotImplementedError

    def aggregate_blocks(self, blocks: list[EncodedContributionBlock], block_plan: BlockPlan) -> EncodedPayloadBlock:
        raise NotImplementedError

    def add_plaintext_to_payload_block(
        self,
        block: EncodedPayloadBlock,
        tensor: torch.Tensor,
        block_plan: BlockPlan,
    ) -> EncodedPayloadBlock:
        raise NotImplementedError

    def decode_payload_block(self, block: EncodedPayloadBlock) -> torch.Tensor:
        raise NotImplementedError

    def serialized_size(self, payload: Any) -> int:
        return 0

    def server_can_decrypt(self) -> bool:
        return False

    def diagnostics(self) -> dict[str, Any]:
        return {
            "feature_pretrain_transport": self.name,
            "homomorphic_encryption": self.is_encrypted,
            "server_can_decrypt": self.server_can_decrypt(),
            "server_can_inspect_plaintext_values": not self.is_encrypted,
        }


class PlainTensorTransport(FeaturePretrainTransport):
    name = "plain"
    is_encrypted = False

    def encode_contribution_block(self, block: FeatureContributionBlock) -> EncodedContributionBlock:
        return EncodedContributionBlock(
            source_client_id=block.source_client_id,
            recipient_client_id=block.recipient_client_id,
            block_id=block.block_id,
            row_ids=block.row_ids.clone(),
            payload=block.contribution.detach().cpu().clone(),
            feature_shape=block.feature_shape,
            transport=self.name,
            serialized_bytes=0,
            is_encrypted=False,
            metadata=dict(block.metadata),
        )

    def aggregate_blocks(self, blocks: list[EncodedContributionBlock], block_plan: BlockPlan) -> EncodedPayloadBlock:
        aggregate = torch.zeros(block_plan.shape, dtype=torch.float)
        for block in blocks:
            aggregate += block.payload.to(aggregate.dtype)
        return EncodedPayloadBlock(
            recipient_client_id=block_plan.recipient_client_id,
            block_id=block_plan.block_id,
            row_ids=block_plan.row_ids.clone(),
            payload=aggregate,
            feature_shape=block_plan.shape,
            transport=self.name,
            serialized_bytes=0,
            is_encrypted=False,
        )

    def decode_payload_block(self, block: EncodedPayloadBlock) -> torch.Tensor:
        return block.payload.detach().cpu().clone()

    def add_plaintext_to_payload_block(
        self,
        block: EncodedPayloadBlock,
        tensor: torch.Tensor,
        block_plan: BlockPlan,
    ) -> EncodedPayloadBlock:
        payload = block.payload.detach().cpu().to(torch.float) + tensor.detach().cpu().to(torch.float)
        return EncodedPayloadBlock(
            recipient_client_id=block_plan.recipient_client_id,
            block_id=block_plan.block_id,
            row_ids=block_plan.row_ids.clone(),
            payload=payload,
            feature_shape=block_plan.shape,
            transport=self.name,
            serialized_bytes=0,
            is_encrypted=False,
            metadata=dict(block.metadata),
        )


@dataclass
class TenSEALCKKSTransport(FeaturePretrainTransport):
    """TenSEAL/CKKS transport for local protocol simulation.

    The server-side operations use only the public context. The secret context is
    used for client-side decode in this single-process simulation.
    """

    context_bundle: TenSEALContextBundle | None = None
    config: TenSEALContextConfig | None = None
    name: str = "he"
    is_encrypted: bool = True

    def __post_init__(self) -> None:
        if self.context_bundle is None:
            self.context_bundle = create_tenseal_context_bundle(self.config)

    def encode_contribution_block(self, block: FeatureContributionBlock) -> EncodedContributionBlock:
        values = block.contribution.detach().cpu().reshape(-1).tolist()
        encrypted = ckks_vector(self.context_bundle.secret_context, [float(value) for value in values]).serialize()
        return EncodedContributionBlock(
            source_client_id=block.source_client_id,
            recipient_client_id=block.recipient_client_id,
            block_id=block.block_id,
            row_ids=block.row_ids.clone(),
            payload=encrypted,
            feature_shape=block.feature_shape,
            transport=self.name,
            serialized_bytes=len(encrypted),
            is_encrypted=True,
            metadata=dict(block.metadata),
        )

    def aggregate_blocks(self, blocks: list[EncodedContributionBlock], block_plan: BlockPlan) -> EncodedPayloadBlock:
        if blocks:
            aggregate = ckks_vector_from(self.context_bundle.public_context, blocks[0].payload)
            for block in blocks[1:]:
                aggregate += ckks_vector_from(self.context_bundle.public_context, block.payload)
        else:
            values = [0.0] * int(block_plan.shape[0] * block_plan.shape[1])
            aggregate = ckks_vector(self.context_bundle.public_context, values)
        serialized = aggregate.serialize()
        return EncodedPayloadBlock(
            recipient_client_id=block_plan.recipient_client_id,
            block_id=block_plan.block_id,
            row_ids=block_plan.row_ids.clone(),
            payload=serialized,
            feature_shape=block_plan.shape,
            transport=self.name,
            serialized_bytes=len(serialized),
            is_encrypted=True,
        )

    def decode_payload_block(self, block: EncodedPayloadBlock) -> torch.Tensor:
        decrypted = ckks_vector_from(self.context_bundle.secret_context, block.payload).decrypt()
        return torch.tensor(decrypted, dtype=torch.float).view(block.feature_shape)

    def add_plaintext_to_payload_block(
        self,
        block: EncodedPayloadBlock,
        tensor: torch.Tensor,
        block_plan: BlockPlan,
    ) -> EncodedPayloadBlock:
        aggregate = ckks_vector_from(self.context_bundle.public_context, block.payload)
        values = tensor.detach().cpu().reshape(-1).tolist()
        encrypted_delta = ckks_vector(self.context_bundle.public_context, [float(value) for value in values])
        aggregate += encrypted_delta
        serialized = aggregate.serialize()
        return EncodedPayloadBlock(
            recipient_client_id=block_plan.recipient_client_id,
            block_id=block_plan.block_id,
            row_ids=block_plan.row_ids.clone(),
            payload=serialized,
            feature_shape=block_plan.shape,
            transport=self.name,
            serialized_bytes=len(serialized),
            is_encrypted=True,
            metadata=dict(block.metadata),
        )

    def serialized_size(self, payload: Any) -> int:
        return len(payload)

    def diagnostics(self) -> dict[str, Any]:
        config = self.context_bundle.config
        base = super().diagnostics()
        base.update(
            {
                "he_library": "tenseal",
                "he_scheme": "CKKS",
                "ciphertext_layout": "recipient_aligned_blocks",
                "poly_modulus_degree": config.poly_modulus_degree,
                "coeff_mod_bit_sizes": list(config.coeff_mod_bit_sizes),
                "global_scale": config.global_scale,
                "context_generated_per_run": True,
                "context_public_serialized_bytes": self.context_bundle.public_serialized_bytes,
                "context_secret_serialized_bytes": self.context_bundle.secret_serialized_bytes,
                "secret_key_visible_to_server": False,
                "simulation_single_process": True,
            }
        )
        return base


def make_feature_pretrain_transport(name: str) -> FeaturePretrainTransport:
    if name == "plain":
        return PlainTensorTransport()
    if name == "he":
        return TenSEALCKKSTransport()
    raise ValueError(f"Unsupported feature-pretrain transport: {name}")
