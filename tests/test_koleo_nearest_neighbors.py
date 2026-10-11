# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math

import pytest
import torch
import torch.nn.functional as F

from dinov2.loss import KoLeoLoss


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("direction", [[1.0, 0.0], [0.3, -0.4, 0.5]])
def test_antipodal_points_select_each_other(dtype, direction):
    """Both rows have one valid non-self neighbor even at similarity -1."""
    direction = torch.tensor(direction, dtype=dtype)
    points = F.normalize(torch.stack((direction, -direction)), dim=-1)

    neighbors = KoLeoLoss().pairwise_NNs_inner(points)

    torch.testing.assert_close(neighbors, torch.tensor([1, 0], device=points.device))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("offset", [0.0, 1e-5, 1e-4])
def test_two_point_loss_and_gradient_match_non_self_distances(dtype, offset):
    """KoLeo uses the distance to the other sample, including rounded -1 dot products."""
    points = torch.tensor([[1.0, 0.0], [-1.0, offset]], dtype=dtype, requires_grad=True)
    reference_points = points.detach().clone().requires_grad_(True)

    loss = KoLeoLoss()(points)
    gradient = torch.autograd.grad(loss, points)[0]

    normalized = F.normalize(reference_points, eps=1e-8, dim=-1)
    differences = normalized - normalized.flip(0)
    distances = torch.linalg.vector_norm(differences + 1e-8, dim=-1)
    reference_loss = -torch.log(distances + 1e-8).mean()
    reference_gradient = torch.autograd.grad(reference_loss, reference_points)[0]
    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(loss, reference_loss, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(gradient, reference_gradient, atol=1e-8, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("offset", [0.0, 1e-5, 0.2])
def test_two_point_loss_and_gradient_match_analytic_geometry(dtype, offset):
    """The normalized chord has length 2*cos(atan(offset)/2)."""
    points = torch.tensor([[1.0, 0.0], [-1.0, offset]], dtype=dtype, requires_grad=True)
    loss = KoLeoLoss()(points)
    gradient = torch.autograd.grad(loss, points)[0]

    radius = math.sqrt(1 + offset**2)
    cosine, sine = 1 / radius, offset / radius
    squared_distance = 2 + 2 * cosine
    expected_loss = -math.log(2 * math.cos(math.atan(offset) / 2))
    expected_gradient = torch.tensor(
        [
            [0.0, sine / squared_distance],
            [sine**2 / (radius * squared_distance), cosine * sine / (radius * squared_distance)],
        ],
        dtype=dtype,
    )
    # PairwiseDistance and the final logarithm retain their existing 1e-8 epsilons.
    tolerance = 2e-7 if dtype == torch.float32 else 2e-8
    torch.testing.assert_close(loss, torch.tensor(expected_loss, dtype=dtype), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(gradient, expected_gradient, atol=1e-8, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ordinary_neighbors_remain_nearest_non_self_rows(dtype):
    """Masking the diagonal preserves ordinary orthogonal and negative neighbors."""
    points = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=dtype), dim=-1)
    neighbors = KoLeoLoss().pairwise_NNs_inner(points)
    torch.testing.assert_close(neighbors, torch.tensor([1, 0, 0]))


def test_coincident_distinct_samples_are_valid_neighbors():
    """Equal features at separate row indices remain valid nearest neighbors."""
    points = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    neighbors = KoLeoLoss().pairwise_NNs_inner(points)
    torch.testing.assert_close(neighbors, torch.tensor([1, 0]))


def test_singleton_behavior_is_unchanged():
    """A singleton has no other sample; this change preserves its existing fallback."""
    points = torch.tensor([[1.0, 0.0]])
    loss = KoLeoLoss()
    torch.testing.assert_close(loss.pairwise_NNs_inner(points), torch.tensor([0]))
    assert torch.isfinite(loss(points))
