"""Side-by-side compare: hands-zero vs hands-estimated for the same frames."""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
import numpy as np, torch, cv2, smplx
from natsort import natsorted

SMPL_ROOT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), 'resource')


def project_and_shade(verts, faces, K, out_size):
    H, W = out_size
    img_W, img_H = int(K[0, 2] * 2), int(K[1, 2] * 2)
    sx, sy = W / img_W, H / img_H
    K2 = K.copy(); K2[0, 0] *= sx; K2[0, 2] *= sx; K2[1, 1] *= sy; K2[1, 2] *= sy
    z = np.maximum(verts[:, 2], 1e-3)
    u = K2[0, 0] * verts[:, 0] / z + K2[0, 2]
    v = K2[1, 1] * verts[:, 1] / z + K2[1, 2]
    fv3 = verts[faces]; fu = u[faces]; fv = v[faces]; fz = z[faces].mean(1)
    e1 = fv3[:, 1] - fv3[:, 0]; e2 = fv3[:, 2] - fv3[:, 0]
    n = np.cross(e1, e2); n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    light = np.array([0.3, -0.7, -0.7]); light /= np.linalg.norm(light)
    intensity = np.clip(n @ light, 0.15, 1.0)
    order = np.argsort(-fz)
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    base = np.array([200, 200, 215], dtype=np.float32)
    for k in order:
        pts = np.stack([fu[k], fv[k]], axis=1)
        if (pts[:, 0] < -5).all() or (pts[:, 0] > W + 5).all(): continue
        if (pts[:, 1] < -5).all() or (pts[:, 1] > H + 5).all(): continue
        col = tuple(int(c) for c in (base * intensity[k]).clip(0, 255))
        cv2.fillConvexPoly(img, pts.astype(np.int32), col, lineType=cv2.LINE_AA)
    return img


def render_frame(model, faces, p, K, cell, device):
    def t(x): return torch.tensor(x, dtype=torch.float32, device=device).view(1, -1)
    with torch.no_grad():
        out = model(
            global_orient=t(p['root_pose']), body_pose=t(p['body_pose']),
            jaw_pose=t(p['jaw_pose']),
            leye_pose=t(p['leye_pose']), reye_pose=t(p['reye_pose']),
            left_hand_pose=t(p['lhand_pose']), right_hand_pose=t(p['rhand_pose']),
            expression=t(p['expr']), transl=t(p['trans']),
            betas=torch.zeros((1, 10), device=device),
        )
    v = out.vertices[0].cpu().numpy()
    return project_and_shade(v, faces, K, (cell, cell))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_zero', required=True)
    ap.add_argument('--root_hands', required=True)
    ap.add_argument('--frames', nargs='+', type=int, default=[0, 96, 145, 290, 387, 484])
    ap.add_argument('--cell', type=int, default=380)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = smplx.SMPLX(model_path=os.path.join(SMPL_ROOT, 'smplx'),
                       gender='neutral', num_betas=10, num_expression_coeffs=10,
                       use_face_contour=True, use_pca=False, flat_hand_mean=True,
                       ext='pkl', batch_size=1).to(device)
    faces = model.faces.astype(np.int32)

    def load_frame(root, i):
        with open(os.path.join(root, 'smplx_params_smoothed', f'{i}.json')) as f: p = json.load(f)
        with open(os.path.join(root, 'cam_params', f'{i}.json')) as f: c = json.load(f)
        K = np.eye(3); K[0, 0], K[1, 1] = c['focal']; K[0, 2], K[1, 2] = c['princpt']
        return p, K

    cell = args.cell; pad = 4; label_w = 130; header_h = 30
    rows = 2   # zero / hands
    cols = len(args.frames)
    canvas = np.full((header_h + rows*(cell+pad)+pad, label_w + cols*(cell+pad)+pad, 3),
                     255, dtype=np.uint8)
    # Column headers
    for ci, fid in enumerate(args.frames):
        x = label_w + ci*(cell+pad) + pad
        cv2.putText(canvas, f'#{fid}', (x+8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    for ri, (lbl, root) in enumerate([('hands=0', args.root_zero),
                                       ('hands=est', args.root_hands)]):
        y = header_h + ri*(cell+pad) + pad
        cv2.putText(canvas, lbl, (8, y+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        for ci, fid in enumerate(args.frames):
            p, K = load_frame(root, fid)
            img = render_frame(model, faces, p, K, cell, device)
            x = label_w + ci*(cell+pad) + pad
            canvas[y:y+cell, x:x+cell] = img
            print(f'  row={lbl} frame={fid}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, canvas)
    print(f'[viz] wrote {args.out}')


if __name__ == '__main__':
    main()
