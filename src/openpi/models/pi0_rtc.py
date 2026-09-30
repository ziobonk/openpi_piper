"""Real-time chunking sampler for Pi0 and Pi0.5.

This is the Kai0 RTC sampler adapted to the current OpenPI model. It inherits
the ordinary Pi0 parameter tree and loss; only action sampling is changed.
"""

from __future__ import annotations

import einops
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at


def _prefix_weights(start: int, end: int, total: int, schedule: str) -> jax.Array:
    start = jnp.minimum(start, end)
    if schedule == "ones":
        weights = jnp.ones(total)
    elif schedule == "zeros":
        weights = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule in ("linear", "exp"):
        weights = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            weights = weights * jnp.expm1(weights) / (jnp.e - 1)
    else:
        raise ValueError(f"Unsupported RTC prefix schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, weights)


class Pi0RTC(pi0.Pi0):
    """Pi0 sampler with latency-aware guidance toward the previous chunk."""

    def __init__(self, config: pi0_config.Pi0Config, rngs):
        super().__init__(config, rngs)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        prev_action_chunk: at.Float[at.Array, "b ah ad"] | None = None,
        inference_delay: int | None = None,
        execute_horizon: int | None = None,
        rtc_action_dim: int | None = None,
        mask_prefix_delay: bool = False,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 0.5,
        enable_rtc: bool = True,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def velocity(x_t, time):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn, suffix_attn_mask], axis=-1)
            suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # Presence/absence of a previous chunk is structural and therefore a
        # valid JIT specialization boundary. The boolean itself remains a JAX
        # value so guidance can still be disabled without Python tracer checks.
        rtc_enabled = jnp.asarray(enable_rtc)
        use_rtc = prev_action_chunk is not None
        if use_rtc:
            previous = jnp.asarray(prev_action_chunk, dtype=noise.dtype)
            if previous.ndim == 2:
                previous = previous[None, ...]
            previous = jnp.nan_to_num(previous)
            if previous.shape[1] < self.action_horizon:
                pad_horizon = self.action_horizon - previous.shape[1]
                previous = jnp.pad(previous, ((0, 0), (0, pad_horizon), (0, 0)), mode="edge")
            previous = previous[:, : self.action_horizon]
            if previous.shape[-1] < self.action_dim:
                previous = jnp.pad(previous, ((0, 0), (0, 0), (0, self.action_dim - previous.shape[-1])))
            previous = previous[..., : self.action_dim]

            delay = jnp.clip(jnp.asarray(0 if inference_delay is None else inference_delay), 0, self.action_horizon)
            horizon = jnp.clip(
                jnp.asarray(self.action_horizon if execute_horizon is None else execute_horizon),
                1,
                self.action_horizon,
            )
            constrained_dim = (
                self.action_dim if rtc_action_dim is None else jnp.clip(jnp.asarray(rtc_action_dim), 0, self.action_dim)
            )
            dim_mask = (jnp.arange(self.action_dim) < constrained_dim)[None, None, :]
            weights = _prefix_weights(delay, horizon, self.action_horizon, prefix_attention_schedule)

        def step(carry):
            x_t, time = carry
            if not use_rtc:
                v_t = velocity(x_t, time)
            else:
                x_for_denoise = x_t
                if mask_prefix_delay:
                    delay_mask = (jnp.arange(self.action_horizon) < delay)[None, :, None]
                    x_for_denoise = jnp.where(delay_mask & dim_mask, previous, x_for_denoise)

                def denoiser(x_local):
                    local_velocity = velocity(x_local, time)
                    return x_local - time * local_velocity, local_velocity

                predicted_action, vjp_fn, base_velocity = jax.vjp(denoiser, x_for_denoise, has_aux=True)
                error = (previous - predicted_action) * weights[None, :, None] * dim_mask * rtc_enabled
                correction = vjp_fn(error)[0]
                tau = jnp.clip(1.0 - time, 1e-3, 1.0)
                one_minus_tau_sq = (1.0 - tau) ** 2
                inverse_variance = (one_minus_tau_sq + tau**2) / jnp.maximum(one_minus_tau_sq, 1e-6)
                coefficient = jnp.nan_to_num((1.0 - tau) / tau, posinf=max_guidance_weight)
                guidance = jnp.minimum(coefficient * inverse_variance, max_guidance_weight)
                v_t = base_velocity - guidance * correction
            v_t = jnp.nan_to_num(v_t)
            return x_t + dt * v_t, time + dt

        actions, _ = jax.lax.fori_loop(0, num_steps, lambda _, carry: step(carry), (noise, 1.0))
        return jnp.nan_to_num(actions)
