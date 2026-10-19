"""Observed-motion edge features for GATraj. Velocities use metres per frame."""
import torch
from torch import nn


def motion_geometry(observed, window=3, horizon=12, risk_radius=2.0,
                    velocity_scale=0.5, confidence_gate=True):
    """Return pair features [receiver, sender, feature] using observations only.

    Closest approach is a constant-velocity proxy, not a predicted trajectory.
    Times are in sampled frames; ETH/UCY samples are about 0.4 seconds apart.
    """
    if observed.shape[0] < 2:
        raise ValueError('Motion features require at least two observed positions')
    differences = observed[1:] - observed[:-1]
    recent = differences[-min(window, differences.shape[0]):]
    velocity = recent.mean(dim=0)
    noise = (recent - velocity).square().sum(-1).mean(0).sqrt()
    confidence = 1.0 / (1.0 + noise / (velocity.norm(dim=-1) + 0.1))
    pair_confidence = confidence[:, None] * confidence[None, :]
    if not confidence_gate:
        pair_confidence = torch.ones_like(pair_confidence)

    position = observed[-1]
    relative_position = position[None, :, :] - position[:, None, :]
    relative_velocity = velocity[None, :, :] - velocity[:, None, :]
    distance = relative_position.norm(dim=-1)
    speed_squared = relative_velocity.square().sum(-1)
    dot = (relative_position * relative_velocity).sum(-1)
    raw_time = -dot / speed_squared.clamp_min(1e-8)
    time = raw_time.clamp(min=0, max=horizon)
    closest = (relative_position + time[..., None] * relative_velocity).norm(dim=-1)
    approaching = (dot < 0) & (speed_squared > 1e-8)
    risk = torch.exp(-0.5 * (closest / risk_radius).square()) * approaching
    closing_speed = -dot / distance.clamp_min(1e-4)
    velocity_features = torch.tanh(relative_velocity / velocity_scale)
    risk_features = torch.stack((
        time / horizon,
        (closest / risk_radius).clamp(max=4),
        torch.tanh(closing_speed / velocity_scale),
        risk,
    ), dim=-1)
    return velocity_features, risk_features, pair_confidence, {
        'distance': distance, 'closest': closest, 'time': raw_time,
        'approaching': approaching,
    }


class MotionEdgeEncoder(nn.Module):
    """A zero-initialized residual on GATraj's existing relative embedding."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        features = 2 * int(args.relative_velocity) + 4 * int(args.closest_approach)
        if features == 0:
            raise ValueError('Enable relative_velocity and/or closest_approach')
        self.projection = nn.Sequential(
            nn.Linear(features, args.hidden_size), nn.SiLU(),
            nn.Linear(args.hidden_size, args.hidden_size),
        )
        # Start from the shared corrected baseline, then learn the residual.
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, observed, neighbor_mask):
        args = self.args
        velocity, risk, confidence, geometry = motion_geometry(
            observed, args.velocity_window, args.pred_length,
            args.risk_radius, args.velocity_scale, args.confidence_gate,
        )
        features = []
        if args.relative_velocity:
            features.append(velocity)
        if args.closest_approach:
            features.append(risk)
        edge_embedding = self.projection(torch.cat(features, dim=-1))
        edge_embedding = edge_embedding * confidence[..., None]
        mask = neighbor_mask.bool()
        if args.adaptive_neighbors:
            extra = (
                (geometry['distance'] <= args.adaptive_radius)
                & (geometry['closest'] <= args.risk_radius)
                & geometry['approaching']
                & (geometry['time'] <= args.pred_length)
                & (confidence >= 0.5)
            )
            mask = mask | extra
        diagonal = torch.eye(mask.shape[0], device=mask.device, dtype=torch.bool)
        return edge_embedding, mask & ~diagonal
