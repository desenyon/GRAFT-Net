"""Layer-wise auxiliary objectives and detached first-order utility targets."""

import torch
from torch import Tensor

from graft_net.losses.future_prediction import future_prediction_loss
from graft_net.losses.gradient_prediction import gradient_prediction_loss
from graft_net.losses.load_balance import load_balance_loss
from graft_net.losses.topology import topology_loss
from graft_net.models.backbone import BackboneOutput
from graft_net.models.config import GraftNetConfig


def auxiliary_outputs(
    backbone: object,
    task_loss: Tensor,
    cfg: GraftNetConfig,
    training: bool,
    mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Average active objectives across all blocks; baselines have none.

    Utility is minus the dot product of d(task loss)/d(mixture output)
    with each candidate expert output. It estimates a local loss reduction,
    not the result of retraining an expert. Targets are detached and centered,
    then RMS-normalized over experts to avoid dependence on batch reduction.
    """
    zero = task_loss.new_zeros(())
    result = {f"{name}_loss": zero for name in ("future", "grad_routing", "topology", "balance")}
    if not isinstance(backbone, BackboneOutput):
        return result
    blocks = backbone.all_block_outputs
    gradients: tuple[Tensor, ...] = ()
    if training and torch.is_grad_enabled() and cfg.use_gradient_routing and cfg.lambda_grad > 0:
        gradients = torch.autograd.grad(
            task_loss,
            [b.expert_out.output for b in blocks],
            retain_graph=True,
            create_graph=False,
        )
    for index, block in enumerate(blocks):
        future = block.attn_out.future_state
        # Representation prediction within the block, not a future time label.
        target = block.hidden.detach()
        scores = block.expert_out.routing_scores
        adjacency = block.topo_out.soft_adjacency
        routing_target = torch.zeros_like(scores)
        if gradients:
            values = block.expert_out.expert_values
            assert values is not None
            utility = -(gradients[index].detach().float().unsqueeze(-2) * values.float()).sum(-1)
            centered = utility - utility.mean(-1, keepdim=True)
            routing_target = (
                centered / centered.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-8)
            ).detach()
        # Keep last-layer diagnostic keys compatible with the original API.
        result.update(
            future_state=future,
            future_target=target,
            routing_scores=scores,
            routing_targets=routing_target,
            soft_adjacency=adjacency,
        )
        if mask is not None:
            future, target = future[mask], target[mask]
            scores, routing_target = scores[mask].unsqueeze(0), routing_target[mask].unsqueeze(0)
            adjacency = adjacency[mask]
        if cfg.use_predictive_attention:
            result["future_loss"] = result["future_loss"] + future_prediction_loss(
                future, target
            ) / len(blocks)
        if cfg.use_latent_topology:
            result["topology_loss"] = result["topology_loss"] + topology_loss(adjacency) / len(
                blocks
            )
        if cfg.use_gradient_routing:
            result["balance_loss"] = result["balance_loss"] + load_balance_loss(scores) / len(
                blocks
            )
            if gradients:
                result["grad_routing_loss"] = result[
                    "grad_routing_loss"
                ] + gradient_prediction_loss(scores, routing_target) / len(blocks)
    return result
