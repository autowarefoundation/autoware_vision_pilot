import torch
import torch.nn as nn


class AutoDriveHead(nn.Module):
    """
    Detect head for AutoDrive — outputs:
        - Curvature       : float  (1/m)
        - Distance (CIPO) : float  (m)
        - Priority        : binary (0/1)

    TODO: Architecture to be designed.
    Inputs are sppf_out and c2psa_out from AutoDriveBackbone,
    both of shape [B, width[5], H/32, W/32].
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError("AutoDriveHead is not yet implemented.")

    def forward(self, sppf_out: torch.Tensor, c2psa_out: torch.Tensor):
        raise NotImplementedError("AutoDriveHead is not yet implemented.")
