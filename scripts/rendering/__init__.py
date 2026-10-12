"""Rendering utilities for the Jacobian avatar pipeline.

Public API (import from this package):
    from scripts.rendering.render_orbit import render_orbit
    from scripts.rendering.render_canon_seq import render_canon_seq
    from scripts.rendering.render_per_frame_cropped import render_per_frame_cropped
    from scripts.rendering.build_videos import build_videos_from_pngs

All rendering follows the VIZ CKPT RULE (memory feedback_viz_ckpt_rule.md):
    - Mesh_RGB_rgba, P_normal_rgba → All ckpt (`All_iter_0_sampling_5_Avatar.pth`)
    - GS_RGB_rgba                  → GS#1 ckpt (`GS_iter_0_sampling_1_..._Avatar.pth`)
    - Do NOT use Opt_pose (GS#2) for GS renders in shared visualizations.

Dataset: CUSTOM.
"""
"""
"""
