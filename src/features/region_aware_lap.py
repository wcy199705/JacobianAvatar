"""Region-aware face Laplacian for Cascade_Avatar.

Replaces the dense face Laplacian (which couples body/head/hand faces across
region boundaries) with a per-region Laplacian — edges that cross region
labels are dropped before assembly. This decouples smoothing across body
parts so head/hand stay sharp while the body remains smooth.

Affects every loss that consumes `face_laps`:
  semantic_scale_lap, semantic_albedo_lap, semantic_rgb_lap,
  semantic_mean_offset_norm_lap, semantic_scale_offset_lap.

Wraps `Cascade_Avatar.set_face_laplacian`. The original is invoked first
(to populate `self.face_laps`); the per-level Laplacian is then overwritten
with the region-aware variant.
"""
from __future__ import annotations

import torch

from core.avatar import cascade as cad


def _compute_face_laplacian_region_aware(faces: torch.Tensor,
                                          region_labels: torch.Tensor):
    """face-graph Laplacian with cross-region edges excluded.

    Args:
        faces: [F, 3] face vertex indices
        region_labels: [F] integer region ID per face
                       (0=body, 1=head, 2=lhand, 3=rhand)
    """
    F = faces.shape[0]
    device = faces.device

    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    edges, _ = torch.sort(edges, dim=1)

    unique_edges, inverse_indices = torch.unique(edges, dim=0, return_inverse=True)
    face_ids = torch.arange(F, device=device).repeat(3)
    perm = torch.argsort(inverse_indices)
    sorted_inverse = inverse_indices[perm]
    sorted_face_ids = face_ids[perm]
    shared_mask = sorted_inverse[:-1] == sorted_inverse[1:]
    src_faces = sorted_face_ids[:-1][shared_mask]
    dst_faces = sorted_face_ids[1:][shared_mask]

    # Drop edges that cross regions
    same_region = (region_labels[src_faces] == region_labels[dst_faces])
    n_cross = (~same_region).sum().item()
    src_faces = src_faces[same_region]
    dst_faces = dst_faces[same_region]

    idx_row = torch.cat([src_faces, dst_faces])
    idx_col = torch.cat([dst_faces, src_faces])
    values = torch.full((idx_row.shape[0],), -1.0, device=device)

    degrees = torch.zeros(F, device=device)
    if idx_row.numel() > 0:
        degrees.index_add_(0, idx_row,
                            torch.ones(idx_row.shape[0], device=device))

    diag_row = torch.arange(F, device=device)
    diag_col = torch.arange(F, device=device)
    final_indices = torch.stack([
        torch.cat([idx_row, diag_row]),
        torch.cat([idx_col, diag_col]),
    ])
    final_values = torch.cat([values, degrees])

    L = torch.sparse_coo_tensor(final_indices, final_values, (F, F))
    print(f'[REGION_AWARE_LAP] level F={F}: {n_cross} cross-region edges removed '
          '(per-region independent Laplacian)')
    return L.coalesce()


def apply() -> None:
    orig_set = cad.Cascade_Avatar.set_face_laplacian

    def patched_set(self):
        orig_set(self)
        for level, mesh in enumerate(self.mesh_avatars):
            faces = mesh.face
            num_faces = faces.shape[0]
            avatar_label = self.avatar_indices[level]
            region = torch.zeros(num_faces, dtype=torch.long, device=faces.device)
            region[avatar_label.head] = 1
            region[avatar_label.lhand] = 2
            region[avatar_label.rhand] = 3
            self.face_laps[level] = _compute_face_laplacian_region_aware(faces, region)
        print('[REGION_AWARE_LAP] Cascade_Avatar.face_laps replaced with '
              'region-aware Laplacians (body/head/lhand/rhand independent)')

    cad.Cascade_Avatar.set_face_laplacian = patched_set
