"""
Optical Flow loss visualization tool.

Called from inside the optimizer to check that the OF_init and OF_reg losses work correctly.
Usage: call as below in the OF loss computation of avatar_optimizer.py:

    from core.debug.visualize_of import visualize_optical_flow
    visualize_optical_flow(
        save_dir=debug_folder,
        step=step,
        mesh_init_flow=mesh_init_flow,       # [N, H, W, 2] rendered displacement
        gt_waft_of=tmp_waft_of,               # [N, 2, H, W] ground truth OF
        waft_loss=waft_loss,                  # OFLoss object (for refined flow)
        waft_features=tmp_waft_feature,       # [N, ...] WAFT features (optional)
        masks=tmp_masks,                      # [N, H, W, 1]
        rendered_mask=F_P_mesh_rendered['mask'],
        of_cur_idxs=of_cur_idxs,
        of_next_idxs=of_next_idxs,
        images=tmp_images,                    # [N, H, W, 3] (optional, for context)
    )
"""

import torch
import numpy as np
import cv2
import os


def flow_to_color(flow_np, max_flow=None):
    """Convert optical flow [H, W, 2] to color image [H, W, 3] (BGR)."""
    h, w, _ = flow_np.shape
    fx, fy = flow_np[:, :, 0], flow_np[:, :, 1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    ang = np.arctan2(fy, fx) + np.pi

    if max_flow is None:
        max_flow = max(mag.max(), 1e-5)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip(mag / max_flow * 255, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def save_rgba(path, bgr, mask=None):
    """Save as RGBA PNG. bgr: [H,W,3] uint8. mask: [H,W] float [0,1] or uint8 — alpha channel."""
    if mask is None:
        alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    elif mask.dtype != np.uint8:
        alpha = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    else:
        alpha = mask
    bgra = np.dstack([bgr, alpha])
    cv2.imwrite(path, bgra)


def warp_with_flow(image, flow):
    """Warp image [1, 3, H, W] with flow [1, 2, H, W]. Returns [H, W, 3] numpy."""
    N, _, H, W = image.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=image.device, dtype=torch.float32),
        torch.arange(W, device=image.device, dtype=torch.float32),
        indexing='ij'
    )
    # flow: [1, 2, H, W] where flow[:,0] = dx, flow[:,1] = dy
    new_x = grid_x.unsqueeze(0) + flow[:, 0]  # [1, H, W]
    new_y = grid_y.unsqueeze(0) + flow[:, 1]  # [1, H, W]

    # Normalize to [-1, 1] for grid_sample
    new_x = 2.0 * new_x / (W - 1) - 1.0
    new_y = 2.0 * new_y / (H - 1) - 1.0

    grid = torch.stack([new_x, new_y], dim=-1)  # [1, H, W, 2]
    warped = torch.nn.functional.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return warped[0].permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def visualize_optical_flow(
    save_dir,
    step,
    mesh_init_flow,       # [N, H, W, 2] rendered displacement
    gt_waft_of,           # [N, 2, H, W] ground truth OF
    masks,                # [N, H, W, 1]
    rendered_mask,        # [N, H, W, 1]
    of_cur_idxs,
    of_next_idxs,
    waft_loss=None,       # OFLoss object (has model for refined flow)
    waft_features=None,   # [N, ...] WAFT features
    images=None,          # [N, H, W, 3] or [N, 3, H, W]
    vis_idx=0,            # which batch element to visualize
):
    """
    Save visualization comparing:
    1. GT optical flow (from WAFT pre-extraction)
    2. Rendered mesh displacement (init_flow)
    3. Refined flow (from WAFT network, if available)
    4. Warped images using each flow
    5. Error maps
    """
    out_dir = os.path.join(save_dir, 'of_debug')
    os.makedirs(out_dir, exist_ok=True)

    # --- Prepare flows ---
    # mesh_init_flow: [N, H, W, 2] -> permute to [N, 2, H, W]
    init_flow_perm = mesh_init_flow.permute(0, 3, 1, 2)  # [N, 2, H, W]

    # Get current frame indices
    cur_idx = of_cur_idxs
    nxt_idx = of_next_idxs

    # Masks
    cur_mask = masks[cur_idx]  # [n, H, W, 1]
    if rendered_mask.shape[0] == masks.shape[0]:
        cur_rmask = rendered_mask[cur_idx].detach()
    else:
        cur_rmask = rendered_mask.detach()
    overlap_mask = (cur_mask * cur_rmask).permute(0, 3, 1, 2)  # [n, 1, H, W]

    # Select specific batch element
    gt_of_vis = gt_waft_of[cur_idx][vis_idx]           # [2, H, W]
    init_of_vis = init_flow_perm[vis_idx]               # [2, H, W] (already indexed by cur)
    mask_vis = overlap_mask[vis_idx]                     # [1, H, W]
    H, W = gt_of_vis.shape[1], gt_of_vis.shape[2]

    # --- Flow color maps ---
    # Find shared max flow for consistent coloring
    all_flows = torch.cat([gt_of_vis.reshape(-1), init_of_vis.reshape(-1)])
    max_flow = all_flows.abs().max().item()

    gt_of_np = gt_of_vis.permute(1, 2, 0).cpu().numpy()          # [H, W, 2]
    init_of_np = init_of_vis.permute(1, 2, 0).cpu().numpy()      # [H, W, 2]
    mask_np = mask_vis[0].cpu().numpy()                           # [H, W]

    gt_color = flow_to_color(gt_of_np, max_flow)
    init_color = flow_to_color(init_of_np, max_flow)

    # Masked versions
    mask_3ch = np.stack([mask_np] * 3, axis=-1)
    gt_color_masked = (gt_color * mask_3ch).astype(np.uint8)
    init_color_masked = (init_color * mask_3ch).astype(np.uint8)

    # --- Error map (L1 difference, normalized) ---
    diff = np.abs(gt_of_np - init_of_np)
    diff[:, :, 0] /= max(W, 1)
    diff[:, :, 1] /= max(H, 1)
    error_mag = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2) * mask_np
    error_vis = (np.clip(error_mag / max(error_mag.max(), 1e-5), 0, 1) * 255).astype(np.uint8)
    error_heatmap = cv2.applyColorMap(error_vis, cv2.COLORMAP_JET)

    # Save individual images (BGR + RGBA)
    prefix = f'step{step:04d}'
    cv2.imwrite(os.path.join(out_dir, f'{prefix}_1_gt_flow.png'), gt_color_masked)
    cv2.imwrite(os.path.join(out_dir, f'{prefix}_2_init_flow.png'), init_color_masked)
    cv2.imwrite(os.path.join(out_dir, f'{prefix}_3_error_map.png'), error_heatmap)
    cv2.imwrite(os.path.join(out_dir, f'{prefix}_4_mask.png'), (mask_np * 255).astype(np.uint8))
    # RGBA versions (alpha = mask)
    save_rgba(os.path.join(out_dir, f'{prefix}_1_gt_flow_rgba.png'), gt_color, mask_np)
    save_rgba(os.path.join(out_dir, f'{prefix}_2_init_flow_rgba.png'), init_color, mask_np)

    # --- Refined flow (if WAFT model available) ---
    refined_color_masked = None
    if waft_loss is not None and hasattr(waft_loss, 'model') and waft_loss.model is not None:
        try:
            if waft_features is not None:
                cur_feat = waft_features[cur_idx][vis_idx:vis_idx+1]
                nxt_feat = waft_features[nxt_idx][vis_idx:vis_idx+1]
                input_flow = init_flow_perm[vis_idx:vis_idx+1]
                output = waft_loss.model.forward_w_init_flow_prefeat(
                    cur_feat, nxt_feat, input_flow, iters=2
                )
                refined_flow = output['flow'][-1][0]  # [2, H, W]
            elif images is not None:
                imgs = images if images.shape[1] == 3 else images.permute(0, 3, 1, 2)
                cur_img = imgs[cur_idx][vis_idx:vis_idx+1]
                nxt_img = imgs[nxt_idx][vis_idx:vis_idx+1]
                input_flow = init_flow_perm[vis_idx:vis_idx+1]
                output = waft_loss.model.forward_w_init_flow(
                    cur_img, nxt_img, input_flow, iters=3
                )
                refined_flow = output['flow'][-1][0]  # [2, H, W]
            else:
                refined_flow = None

            if refined_flow is not None:
                refined_np = refined_flow.permute(1, 2, 0).cpu().numpy()
                refined_color = flow_to_color(refined_np, max_flow)
                refined_color_masked = (refined_color * mask_3ch).astype(np.uint8)
                cv2.imwrite(os.path.join(out_dir, f'{prefix}_5_refined_flow.png'), refined_color_masked)
                save_rgba(os.path.join(out_dir, f'{prefix}_5_refined_flow_rgba.png'), refined_color, mask_np)

                # Refined vs GT error (how close refined is to GT)
                ref_gt_diff = np.abs(refined_np - gt_of_np)
                ref_gt_diff[:, :, 0] /= max(W, 1)
                ref_gt_diff[:, :, 1] /= max(H, 1)
                ref_gt_err = np.sqrt(ref_gt_diff[:, :, 0]**2 + ref_gt_diff[:, :, 1]**2) * mask_np
                ref_gt_err_vis = (np.clip(ref_gt_err / max(ref_gt_err.max(), 1e-5), 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(out_dir, f'{prefix}_6_refined_vs_gt_error.png'),
                           cv2.applyColorMap(ref_gt_err_vis, cv2.COLORMAP_JET))
        except Exception as e:
            print(f"[OF_VIS] Refined flow visualization failed: {e}")

    # --- Warped images (if available) ---
    if images is not None:
        imgs = images if images.shape[1] == 3 else images.permute(0, 3, 1, 2)
        cur_img = imgs[cur_idx][vis_idx:vis_idx+1].float()  # [1, 3, H, W]
        nxt_img = imgs[nxt_idx][vis_idx:vis_idx+1].float()

        # Warp next→curr using GT flow
        warped_gt = warp_with_flow(nxt_img, gt_of_vis[None])
        # Warp next→curr using rendered init flow
        warped_init = warp_with_flow(nxt_img, init_of_vis[None])

        cur_np = cur_img[0].permute(1, 2, 0).cpu().numpy()
        nxt_np = nxt_img[0].permute(1, 2, 0).cpu().numpy()

        # Dataset returns BGR (cv2.imread, no cvtColor) — write BGR directly
        cur_bgr = (np.clip(cur_np, 0, 1) * 255).astype(np.uint8)
        nxt_bgr = (np.clip(nxt_np, 0, 1) * 255).astype(np.uint8)
        warped_gt_bgr = (np.clip(warped_gt, 0, 1) * 255).astype(np.uint8)
        warped_init_bgr = (np.clip(warped_init, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f'{prefix}_7_curr_image.png'), cur_bgr)
        cv2.imwrite(os.path.join(out_dir, f'{prefix}_8_next_image.png'), nxt_bgr)
        cv2.imwrite(os.path.join(out_dir, f'{prefix}_9_warped_gt_flow.png'), warped_gt_bgr)
        cv2.imwrite(os.path.join(out_dir, f'{prefix}_10_warped_init_flow.png'), warped_init_bgr)
        # RGBA versions
        save_rgba(os.path.join(out_dir, f'{prefix}_7_curr_image_rgba.png'), cur_bgr, mask_np)
        save_rgba(os.path.join(out_dir, f'{prefix}_8_next_image_rgba.png'), nxt_bgr, mask_np)
        save_rgba(os.path.join(out_dir, f'{prefix}_9_warped_gt_flow_rgba.png'), warped_gt_bgr, mask_np)
        save_rgba(os.path.join(out_dir, f'{prefix}_10_warped_init_flow_rgba.png'), warped_init_bgr, mask_np)

        # Warp next→curr using refined flow (if available)
        if refined_color_masked is not None and refined_flow is not None:
            warped_refined = warp_with_flow(nxt_img, refined_flow[None])
            warped_refined_bgr = (np.clip(warped_refined, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, f'{prefix}_11_warped_refined_flow.png'), warped_refined_bgr)
            save_rgba(os.path.join(out_dir, f'{prefix}_11_warped_refined_flow_rgba.png'), warped_refined_bgr, mask_np)

    # --- Summary panel ---
    panels = [gt_color_masked, init_color_masked, error_heatmap]
    labels = ['GT Flow', 'Rendered Flow', 'Error (init vs GT)']
    if refined_color_masked is not None:
        panels.append(refined_color_masked)
        labels.append('Refined Flow')
        ref_gt_heatmap = cv2.applyColorMap(ref_gt_err_vis, cv2.COLORMAP_JET)
        panels.append(ref_gt_heatmap)
        labels.append('Error (refined vs GT)')

    # Resize all to same size
    target_h = min(p.shape[0] for p in panels)
    target_w = min(p.shape[1] for p in panels)
    panels_resized = [cv2.resize(p, (target_w, target_h)) for p in panels]

    # Add labels
    for i, (panel, label) in enumerate(zip(panels_resized, labels)):
        cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    summary = np.concatenate(panels_resized, axis=1)
    cv2.imwrite(os.path.join(out_dir, f'{prefix}_summary.png'), summary)

    print(f"[OF_VIS] Step {step}: saved to {out_dir}")
