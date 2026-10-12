"""Subprocess bridge: runs MHR conversion in the pixi Python 3.12 env
where pymomentum is available. Called by smpl2mhr.py via subprocess."""
import sys
import os

# Remove script directory from sys.path to avoid shadowing smplx package
# with utils/io/smplx.py
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != script_dir]

import torch

def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    data = torch.load(input_path, map_location="cpu", weights_only=False)
    smpl_config = data["smpl_config"]
    smpl_full_pose = data["smpl_full_pose"]
    lod = data["lod"]

    # libs/MHR is the unmodified upstream repo: `mhr` is importable through the pixi env,
    # the SMPL(-X) conversion tool is a script folder that loads its assets relative to cwd.
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    conv_dir = os.path.join(project_root, 'libs', 'MHR', 'tools', 'mhr_smpl_conversion')
    sys.path.insert(0, conv_dir)

    import smplx
    assert hasattr(smplx, 'SMPLX'), f"smplx module at {smplx.__file__} has no SMPLX attribute"
    from mhr.mhr import MHR
    import file_assets
    for _name in [n for n in dir(file_assets) if n.endswith('_FILE')]:
        _path = getattr(file_assets, _name)
        if not os.path.isabs(_path):
            setattr(file_assets, _name, os.path.normpath(os.path.join(conv_dir, _path)))
    from conversion import Conversion

    device = torch.device("cpu")
    mhr_model = MHR.from_files(device=device, lod=lod)
    smplx_model = smplx.SMPLX(
        model_path="resource/smplx",
        gender=smpl_config["gender"],
        num_pca_comps=smpl_config["num_pca_comp"],
    )

    converter = Conversion(
        mhr_model=MHR.from_files(device=device, lod=1),
        smpl_model=smplx_model,
        method="pymomentum",
    )
    results = converter.convert_smpl2mhr(
        smpl_parameters=smpl_full_pose,
        single_identity=True,
        return_mhr_meshes=True,
        return_mhr_parameters=True,
    )
    identity_coeffs = results.result_parameters["identity_coeffs"]
    lbs_model_params = results.result_parameters["lbs_model_params"]
    face_expr_coeffs = results.result_parameters["face_expr_coeffs"]
    vertices, _ = mhr_model(identity_coeffs, lbs_model_params, face_expr_coeffs)
    faces = mhr_model.character.mesh.faces

    faces_out = torch.from_numpy(faces) if isinstance(faces, __import__('numpy').ndarray) else faces.cpu()
    torch.save({"vertices": vertices.cpu(), "faces": faces_out}, output_path)

if __name__ == "__main__":
    main()
