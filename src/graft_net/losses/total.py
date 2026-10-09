"""Total combined loss for GRAFT-Net training.

L_total = L_task
        + λ_future  * L_future_prediction
        + λ_grad    * L_gradient_prediction
        + λ_topology* L_topology
        + λ_balance * L_load_balance
"""

from __future__ import annotations

from torch import Tensor

from graft_net.losses.future_prediction import future_prediction_loss
from graft_net.losses.gradient_prediction import gradient_prediction_loss
from graft_net.losses.load_balance import load_balance_loss
from graft_net.losses.topology import topology_loss


def compute_total_loss(
    outputs: dict[str, Tensor],
    lambda_future: float = 0.1,
    lambda_grad: float = 0.05,
    lambda_topology: float = 0.01,
    lambda_balance: float = 0.01,
) -> dict[str, Tensor]:
    """Compute weighted total loss from model output dict.

    Task models provide task_loss and scalar future_loss, grad_routing_loss,
    topology_loss, balance_loss (averaged over active layers). Legacy dictionaries
    with future_state/target, routing_scores/targets, and soft_adjacency still work.
    Baselines and disabled mechanisms provide zero auxiliary losses.

    Returns:
        dict with 'total' plus individual named components.
    """
    losses: dict[str, Tensor] = {}

    losses["task"] = outputs["task_loss"]

    losses["future"] = (
        outputs["future_loss"]
        if "future_loss" in outputs
        else future_prediction_loss(
            outputs["future_state"],
            outputs["future_target"],
        )
    )

    losses["grad_routing"] = (
        outputs["grad_routing_loss"]
        if "grad_routing_loss" in outputs
        else gradient_prediction_loss(
            outputs["routing_scores"],
            outputs["routing_targets"],
        )
    )

    losses["topology"] = (
        outputs["topology_loss"]
        if "topology_loss" in outputs
        else topology_loss(outputs["soft_adjacency"])
    )

    losses["balance"] = (
        outputs["balance_loss"]
        if "balance_loss" in outputs
        else load_balance_loss(outputs.get("expert_probs", outputs["routing_scores"]))
    )

    losses["total"] = (
        losses["task"]
        + lambda_future * losses["future"]
        + lambda_grad * losses["grad_routing"]
        + lambda_topology * losses["topology"]
        + lambda_balance * losses["balance"]
    )

    return losses
