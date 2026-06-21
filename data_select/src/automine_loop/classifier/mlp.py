from __future__ import annotations

from automine_loop.classifier.linear_probe import predict_probabilities, train_linear_probe


def train_mlp(*args, **kwargs):
    """Fallback MLP entrypoint.

    The MVP uses a linear probe for reproducibility. This function preserves the
    planned module boundary so a torch MLP can be dropped in later.
    """

    return train_linear_probe(*args, **kwargs)
