import os
import subprocess
import tempfile
import torch

# Path to the pixi Python 3.12 env that has pymomentum
_PIXI_PYTHON = os.path.join(
    os.path.dirname(__file__), "..", "..", "libs", "MHR", ".pixi",
    "envs", "default", "bin", "python",
)
_BRIDGE_SCRIPT = os.path.join(os.path.dirname(__file__), "_mhr_bridge.py")
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def convert2MHR(smpl_config, smpl_full_pose, lod):
    """Run MHR conversion via subprocess using the pixi env (Python 3.12)
    where pymomentum is available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.pt")
        output_path = os.path.join(tmpdir, "output.pt")

        torch.save({
            "smpl_config": smpl_config,
            "smpl_full_pose": smpl_full_pose.cpu() if hasattr(smpl_full_pose, 'cpu') else smpl_full_pose,
            "lod": lod,
        }, input_path)

        pixi_env_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "libs", "MHR", ".pixi", "envs", "default",
        ))
        env = os.environ.copy()
        # Don't set PYTHONPATH - the bridge script appends project root itself
        # to avoid shadowing the real smplx package with utils/io/smplx.py
        env.pop("PYTHONPATH", None)
        env["LD_LIBRARY_PATH"] = os.path.join(pixi_env_root, "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        # Ensure pixi env's PATH is first so its python and libs take precedence
        env["PATH"] = os.path.join(pixi_env_root, "bin") + ":" + env.get("PATH", "")

        result = subprocess.run(
            [os.path.abspath(_PIXI_PYTHON), _BRIDGE_SCRIPT, input_path, output_path],
            cwd=os.path.abspath(_PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"MHR bridge subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        data = torch.load(output_path, map_location="cuda", weights_only=False)
        return data["vertices"], data["faces"]
