from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.TSPEnv import Step_State


@dataclass
class ModelConfig:
    node_feature_dim: int = 6
    embedding_dim: int = 128
    encoder_layer_num: int = 6
    qkv_dim: int = 16
    head_num: int = 8
    logit_clipping: float = 10.0
    ff_hidden_dim: int = 512
    activation: str = "relu"
    dropout: float = 0.0
    block_size: int = 1
    max_pomo_size: int = 8
    pomo_divisor: int | None = None
    start_node_strategy: str = "spread"


class TSPModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = TSPEncoder(config)
        self.decoder = TSPDecoder(config)
        self.encoded_nodes: torch.Tensor | None = None

    def pre_forward(self, node_features: torch.Tensor) -> None:
        self.encoded_nodes = self.encoder(node_features)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(
        self,
        state: "Step_State",
        decode_type: str = "sample",
        use_pomo_start: bool = True,
        return_all_probs: bool = False,
        return_all_logits: bool = False,
    ):
        if self.encoded_nodes is None:
            raise RuntimeError("call pre_forward before decoding")

        batch_size = state.batch_idx.size(0)
        pomo_size = state.batch_idx.size(1)
        device = self.encoded_nodes.device

        if state.selected_count == 0:
            selected = torch.zeros((batch_size, pomo_size), dtype=torch.long, device=device)
            prob = torch.ones((batch_size, pomo_size), device=device)
            if return_all_probs:
                all_probs = F.one_hot(selected, num_classes=self.encoded_nodes.size(1)).to(dtype=prob.dtype)
                if return_all_logits:
                    return selected, prob, all_probs, torch.zeros_like(all_probs)
                return selected, prob, all_probs
            return selected, prob

        if state.selected_count == 1 and use_pomo_start:
            if state.start_nodes is None:
                raise RuntimeError("start_nodes must be provided when use_pomo_start=True")
            selected = state.start_nodes
            prob = torch.ones((batch_size, pomo_size), device=device)
            encoded_first = _get_encoding(self.encoded_nodes, selected)
            self.decoder.set_q1(encoded_first)
            if return_all_probs:
                all_probs = F.one_hot(selected, num_classes=self.encoded_nodes.size(1)).to(dtype=prob.dtype)
                if return_all_logits:
                    return selected, prob, all_probs, torch.zeros_like(all_probs)
                return selected, prob, all_probs
            return selected, prob

        encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
        if return_all_logits:
            probs, unmasked_logits = self.decoder(
                encoded_last_node,
                ninf_mask=state.ninf_mask,
                return_unmasked_logits=True,
            )
        else:
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)

        if decode_type == "sample":
            while True:
                selected = (
                    probs.reshape(batch_size * pomo_size, -1)
                    .multinomial(1)
                    .squeeze(1)
                    .reshape(batch_size, pomo_size)
                )
                prob = probs[state.batch_idx, state.pomo_idx, selected].reshape(batch_size, pomo_size)
                if (prob > 0).all():
                    break
        elif decode_type == "greedy":
            selected = probs.argmax(dim=2)
            prob = probs[state.batch_idx, state.pomo_idx, selected].reshape(batch_size, pomo_size)
        else:
            raise ValueError(f"unsupported decode_type '{decode_type}'")

        if state.selected_count == 1 and not use_pomo_start:
            encoded_first = _get_encoding(self.encoded_nodes, selected)
            self.decoder.set_q1(encoded_first)
        if return_all_probs:
            if return_all_logits:
                return selected, prob, probs, unmasked_logits
            return selected, prob, probs
        return selected, prob

    def block_logits(
        self,
        state: "Step_State",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ordered position-slot logits over the currently unvisited nodes.

        Shapes are ``[B, POMO, K, R]`` for logits, ``[B, POMO, R]`` for
        the corresponding raw node ids, and ``[K]`` for the active-slot mask.
        ``R`` is identical across rollouts because every rollout advances by
        the same number of nodes, although the actual available node ids may
        differ.  Slots after ``min(K, R)`` are inactive in the final block.
        """

        if self.encoded_nodes is None:
            raise RuntimeError("call pre_forward before decoding")
        if state.current_node is None or state.ninf_mask is None:
            raise RuntimeError("block decoding requires a current node and visited mask")
        encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
        return self.decoder.forward_block(encoded_last_node, state.ninf_mask)

    def forward_block(
        self,
        state: "Step_State",
        decode_type: str = "sample",
        use_pomo_start: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate one ordered block through conflict-free assignment.

        The expensive decoder forward pass produces all ``K x R`` logits at
        once.  A lightweight autoregressive matching then selects slot-node
        pairs without replacement.  Returned nodes and probabilities are
        ordered by slot id, so appending them preserves the CVTSP tour order.
        """

        logits, available_nodes, active_slot_mask = self.block_logits(state)
        selected, selected_probabilities = sample_block_assignment(
            logits,
            available_nodes,
            active_slot_mask,
            decode_type=decode_type,
        )
        if (
            state.selected_count == 1
            and not use_pomo_start
            and self.decoder.q_first is None
            and selected.size(2) > 0
        ):
            self.decoder.set_q1(_get_encoding(self.encoded_nodes, selected[:, :, 0]))
        return selected, selected_probabilities, logits, available_nodes, active_slot_mask


def _get_encoding(encoded_nodes: torch.Tensor, node_index_to_pick: torch.Tensor) -> torch.Tensor:
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    return encoded_nodes.gather(dim=1, index=gathering_index)


class TSPEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.node_feature_dim, config.embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layer_num)])

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        out = self.embedding(node_features)
        for layer in self.layers:
            out = layer(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wq = nn.Linear(config.embedding_dim, config.head_num * config.qkv_dim, bias=False)
        self.wk = nn.Linear(config.embedding_dim, config.head_num * config.qkv_dim, bias=False)
        self.wv = nn.Linear(config.embedding_dim, config.head_num * config.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(config.head_num * config.qkv_dim, config.embedding_dim)
        self.add_and_norm_1 = AddAndNormalization(config.embedding_dim)
        self.feed_forward = FeedForward(
            config.embedding_dim,
            config.ff_hidden_dim,
            activation=config.activation,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.add_and_norm_2 = AddAndNormalization(config.embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        q = reshape_by_heads(self.wq(inputs), self.config.head_num)
        k = reshape_by_heads(self.wk(inputs), self.config.head_num)
        v = reshape_by_heads(self.wv(inputs), self.config.head_num)
        out_concat = multi_head_attention(q, k, v)
        out = self.dropout(self.multi_head_combine(out_concat))
        out = self.add_and_norm_1(inputs, out)
        ff_out = self.dropout(self.feed_forward(out))
        return self.add_and_norm_2(out, ff_out)


class TSPDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.wq_first = nn.Linear(config.embedding_dim, config.head_num * config.qkv_dim, bias=False)
        self.wq_last = nn.Linear(config.embedding_dim, config.head_num * config.qkv_dim, bias=False)
        self.wk = nn.Linear(config.embedding_dim, config.head_num * config.qkv_dim, bias=False)
        self.wv = nn.Linear(config.embedding_dim, config.head_num * config.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(config.head_num * config.qkv_dim, config.embedding_dim)
        if config.block_size <= 0:
            raise ValueError("block_size must be positive")
        if config.block_size > 1:
            self.slot_embedding = nn.Parameter(
                torch.empty(config.block_size, config.embedding_dim)
            )
            self.wq_slot = nn.Linear(
                config.embedding_dim,
                config.head_num * config.qkv_dim,
                bias=False,
            )
            nn.init.normal_(self.slot_embedding, mean=0.0, std=config.embedding_dim ** -0.5)
        else:
            self.register_parameter("slot_embedding", None)
            self.wq_slot = None
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.single_head_key: torch.Tensor | None = None
        self.q_first: torch.Tensor | None = None

    def set_kv(self, encoded_nodes: torch.Tensor) -> None:
        self.k = reshape_by_heads(self.wk(encoded_nodes), self.config.head_num)
        self.v = reshape_by_heads(self.wv(encoded_nodes), self.config.head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        self.q_first = None

    def set_q1(self, encoded_q1: torch.Tensor) -> None:
        self.q_first = reshape_by_heads(self.wq_first(encoded_q1), self.config.head_num)

    def forward(
        self,
        encoded_last_node: torch.Tensor,
        ninf_mask: torch.Tensor,
        return_unmasked_logits: bool = False,
    ):
        if self.k is None or self.v is None or self.single_head_key is None:
            raise RuntimeError("decoder keys/values not initialized")

        q_last = reshape_by_heads(self.wq_last(encoded_last_node), self.config.head_num)
        q = q_last if self.q_first is None else self.q_first + q_last

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_attn_out = self.multi_head_combine(out_concat)
        score = torch.matmul(mh_attn_out, self.single_head_key)

        sqrt_embedding_dim = self.config.embedding_dim ** 0.5
        score_scaled = score / sqrt_embedding_dim
        score_clipped = self.config.logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        probs = F.softmax(score_masked, dim=2)
        if return_unmasked_logits:
            return probs, score_clipped
        return probs

    def forward_block(
        self,
        encoded_last_node: torch.Tensor,
        ninf_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute one joint position-slot action matrix.

        The logits retain ordered position slots; only the order in which the
        slot-node pairs are sampled is permutation-free.  Therefore this does
        not turn the CVTSP tour into an unordered set.
        """

        if self.k is None or self.v is None or self.single_head_key is None:
            raise RuntimeError("decoder keys/values not initialized")
        if self.slot_embedding is None or self.wq_slot is None:
            raise RuntimeError("forward_block requires ModelConfig.block_size > 1")
        if ninf_mask.ndim != 3:
            raise ValueError("ninf_mask must have shape [B, POMO, N]")

        batch_size, pomo_size, node_count = ninf_mask.shape
        available_mask = torch.isfinite(ninf_mask)
        remaining_counts = available_mask.sum(dim=2)
        if not bool((remaining_counts == remaining_counts[0, 0]).all()):
            raise RuntimeError("all rollouts must have the same remaining-node count")
        remaining = int(remaining_counts[0, 0].item())
        if remaining <= 0:
            raise RuntimeError("cannot decode a block with no remaining nodes")

        node_ids = torch.arange(node_count, device=ninf_mask.device)
        expanded_ids = node_ids.view(1, 1, node_count).expand(batch_size, pomo_size, node_count)
        available_nodes = expanded_ids[available_mask].reshape(batch_size, pomo_size, remaining)

        q_last = reshape_by_heads(self.wq_last(encoded_last_node), self.config.head_num)
        q_base = q_last if self.q_first is None else self.q_first + q_last
        slot_q = self.wq_slot(self.slot_embedding)
        slot_q = slot_q.reshape(
            self.config.block_size,
            self.config.head_num,
            self.config.qkv_dim,
        ).permute(1, 0, 2)
        q = q_base[:, :, :, None, :] + slot_q[None, :, None, :, :]
        q = q.reshape(
            batch_size,
            self.config.head_num,
            pomo_size * self.config.block_size,
            self.config.qkv_dim,
        )
        repeated_mask = ninf_mask[:, :, None, :].expand(
            batch_size,
            pomo_size,
            self.config.block_size,
            node_count,
        ).reshape(batch_size, pomo_size * self.config.block_size, node_count)
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=repeated_mask)
        mh_attn_out = self.multi_head_combine(out_concat).reshape(
            batch_size,
            pomo_size,
            self.config.block_size,
            self.config.embedding_dim,
        )
        dense_score = torch.einsum("bpke,ben->bpkn", mh_attn_out, self.single_head_key)
        dense_score = self.config.logit_clipping * torch.tanh(
            dense_score / (self.config.embedding_dim ** 0.5)
        )
        gather_index = available_nodes[:, :, None, :].expand(
            batch_size,
            pomo_size,
            self.config.block_size,
            remaining,
        )
        logits = dense_score.gather(dim=3, index=gather_index)
        active_slot_mask = (
            torch.arange(self.config.block_size, device=ninf_mask.device)
            < min(self.config.block_size, remaining)
        )
        return logits, available_nodes, active_slot_mask


def sample_block_assignment(
    logits: torch.Tensor,
    available_nodes: torch.Tensor,
    active_slot_mask: torch.Tensor,
    *,
    decode_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a conflict-free slot-node matching and restore slot order."""

    if logits.ndim != 4:
        raise ValueError("logits must have shape [B, POMO, K, R]")
    batch_size, pomo_size, slot_count, remaining = logits.shape
    if available_nodes.shape != (batch_size, pomo_size, remaining):
        raise ValueError("available_nodes shape does not match logits")
    if active_slot_mask.shape != (slot_count,):
        raise ValueError("active_slot_mask must have shape [K]")
    active_count = int(active_slot_mask.sum().item())
    if active_count <= 0 or active_count > remaining:
        raise ValueError("invalid number of active block slots")
    normalized_decode = str(decode_type).strip().lower()
    if normalized_decode not in {"sample", "greedy"}:
        raise ValueError(f"unsupported decode_type '{decode_type}'")

    pair_mask = active_slot_mask.view(1, 1, slot_count, 1).expand(
        batch_size, pomo_size, slot_count, remaining
    ).clone()
    selected_by_slot = torch.full(
        (batch_size, pomo_size, slot_count),
        -1,
        dtype=torch.long,
        device=logits.device,
    )
    probability_by_slot = torch.ones(
        (batch_size, pomo_size, slot_count),
        dtype=logits.dtype,
        device=logits.device,
    )

    for _ in range(active_count):
        masked_logits = logits.masked_fill(~pair_mask, float("-inf"))
        probabilities = F.softmax(masked_logits.reshape(batch_size, pomo_size, -1), dim=2)
        flat = probabilities.reshape(batch_size * pomo_size, -1)
        if normalized_decode == "sample":
            flat_index = flat.multinomial(1).squeeze(1).reshape(batch_size, pomo_size)
        else:
            flat_index = probabilities.argmax(dim=2)
        selected_slot = torch.div(flat_index, remaining, rounding_mode="floor")
        selected_position = flat_index.remainder(remaining)
        selected_node = available_nodes.gather(2, selected_position[:, :, None]).squeeze(2)
        selected_probability = probabilities.gather(2, flat_index[:, :, None]).squeeze(2)
        selected_by_slot.scatter_(2, selected_slot[:, :, None], selected_node[:, :, None])
        probability_by_slot.scatter_(
            2,
            selected_slot[:, :, None],
            selected_probability[:, :, None],
        )
        selected_slot_mask = F.one_hot(selected_slot, num_classes=slot_count).bool()[:, :, :, None]
        selected_node_mask = F.one_hot(selected_position, num_classes=remaining).bool()[:, :, None, :]
        pair_mask &= ~selected_slot_mask
        pair_mask &= ~selected_node_mask

    selected = selected_by_slot[:, :, :active_count]
    probabilities = probability_by_slot[:, :, :active_count]
    if bool((selected < 0).any()):
        raise RuntimeError("block assignment left an active slot unmatched")
    return selected, probabilities


def reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    batch_size = qkv.size(0)
    length = qkv.size(1)
    reshaped = qkv.reshape(batch_size, length, head_num, -1)
    return reshaped.transpose(1, 2)


def multi_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rank2_ninf_mask: torch.Tensor | None = None,
    rank3_ninf_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = q.size(0)
    head_num = q.size(1)
    length = q.size(2)
    key_dim = q.size(3)
    input_size = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float32, device=q.device))

    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_size, head_num, length, input_size)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_size, head_num, length, input_size)

    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    out = out.transpose(1, 2)
    return out.reshape(batch_size, length, head_num * key_dim)


class AddAndNormalization(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor) -> torch.Tensor:
        added = input_1 + input_2
        normalized = self.norm(added.transpose(1, 2))
        return normalized.transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, activation: str = "relu"):
        super().__init__()
        self.w1 = nn.Linear(embedding_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, embedding_dim)
        self.activation_name = str(activation).strip().lower()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.w1(inputs)
        if self.activation_name == "gelu":
            hidden = F.gelu(hidden)
        elif self.activation_name == "relu":
            hidden = F.relu(hidden)
        else:
            raise ValueError(f"unsupported activation '{self.activation_name}'")
        return self.w2(hidden)
