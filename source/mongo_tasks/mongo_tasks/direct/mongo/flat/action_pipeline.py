"""Shared policy-action conditioning for training, Play and deployment."""

from __future__ import annotations

import torch


def condition_policy_actions(
    raw_actions: torch.Tensor,
    previous_actions: torch.Tensor,
    current_weight: torch.Tensor,
    max_delta: torch.Tensor,
    use_low_pass: bool,
    use_slew_rate_limit: bool,
) -> torch.Tensor:
    """Apply the deployable first-order low-pass and per-step slew limit."""
    actions = raw_actions
    if use_low_pass:
        actions = previous_actions * (1.0 - current_weight) + raw_actions * current_weight
    if use_slew_rate_limit:
        delta = torch.clamp(actions - previous_actions, -max_delta, max_delta)
        actions = previous_actions + delta
    return actions


class MongoActionProcessor(torch.nn.Module):
    """Stateful deployment copy of the environment's action-conditioning path.

    The exported module is intentionally separate from the policy network: the
    real robot already contributes its measured transport/actuator latency, so
    deployment must reproduce the low-pass and slew limit but must not add the
    simulation-only delay buffer a second time.
    """

    def __init__(
        self,
        action_dim: int,
        leg_action_dim: int,
        leg_current_weight: float,
        wheel_current_weight: float,
        leg_max_delta: float,
        wheel_max_delta: float,
        use_low_pass: bool = True,
        use_slew_rate_limit: bool = True,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        current_weight = torch.full((action_dim,), float(wheel_current_weight))
        current_weight[:leg_action_dim] = float(leg_current_weight)
        max_delta = torch.full((action_dim,), float(wheel_max_delta))
        max_delta[:leg_action_dim] = float(leg_max_delta)
        self.register_buffer("current_weight", current_weight)
        self.register_buffer("max_delta", max_delta)
        self.register_buffer("previous_actions", torch.zeros(batch_size, action_dim))
        self.use_low_pass = bool(use_low_pass)
        self.use_slew_rate_limit = bool(use_slew_rate_limit)

    @torch.jit.export
    def reset(self) -> None:
        """Clear filter state at controller start or episode reset."""
        self.previous_actions.zero_()

    def forward(self, raw_actions: torch.Tensor) -> torch.Tensor:
        if raw_actions.shape != self.previous_actions.shape:
            raise RuntimeError(
                "MongoActionProcessor input shape must match its exported batch/action shape"
            )
        actions = condition_policy_actions(
            raw_actions,
            self.previous_actions,
            self.current_weight,
            self.max_delta,
            self.use_low_pass,
            self.use_slew_rate_limit,
        )
        self.previous_actions.copy_(actions)
        return actions


__all__ = ["MongoActionProcessor", "condition_policy_actions"]
