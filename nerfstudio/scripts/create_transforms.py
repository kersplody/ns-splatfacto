#!/usr/bin/env python
"""Create transforms.json from a COLMAP binary model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from nerfstudio.data.utils.colmap_parsing_utils import qvec2rotmat, read_cameras_binary, read_images_binary
from nerfstudio.utils.rich_utils import CONSOLE


def parse_colmap_camera_params(camera) -> Dict[str, Any]:
    """Parse a COLMAP camera into nerfstudio-style transforms.json intrinsics."""
    out: Dict[str, Any] = {
        "w": camera.width,
        "h": camera.height,
    }
    camera_params = camera.params

    if camera.model == "SIMPLE_PINHOLE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        out["camera_model"] = "OPENCV"
    elif camera.model == "PINHOLE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        out["camera_model"] = "OPENCV"
    elif camera.model == "SIMPLE_RADIAL":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        out["camera_model"] = "OPENCV"
    elif camera.model == "RADIAL":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["p1"] = 0.0
        out["p2"] = 0.0
        out["camera_model"] = "OPENCV"
    elif camera.model == "OPENCV":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        out["camera_model"] = "OPENCV"
    elif camera.model == "OPENCV_FISHEYE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["k3"] = float(camera_params[6])
        out["k4"] = float(camera_params[7])
        out["camera_model"] = "OPENCV_FISHEYE"
    elif camera.model == "SIMPLE_RADIAL_FISHEYE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["k3"] = 0.0
        out["k4"] = 0.0
        out["camera_model"] = "OPENCV_FISHEYE"
    elif camera.model == "RADIAL_FISHEYE":
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["k3"] = 0.0
        out["k4"] = 0.0
        out["camera_model"] = "OPENCV_FISHEYE"
    else:
        raise NotImplementedError(f"{camera.model} camera model is not supported")

    return out


def create_transforms_data(
    model_dir: Path,
    image_dir: str = "./images",
    keep_original_world_coordinate: bool = False,
    use_single_camera_mode: bool = True,
) -> Dict[str, Any]:
    """Create transforms.json data from COLMAP binary files."""
    cameras_bin = model_dir / "cameras.bin"
    images_bin = model_dir / "images.bin"
    if not cameras_bin.exists() or not images_bin.exists():
        raise FileNotFoundError(f"Expected {cameras_bin} and {images_bin} to exist")

    cam_id_to_camera = read_cameras_binary(cameras_bin)
    im_id_to_image = read_images_binary(images_bin)

    if set(cam_id_to_camera.keys()) != {1}:
        use_single_camera_mode = False
        out: Dict[str, Any] = {}
    else:
        out = parse_colmap_camera_params(cam_id_to_camera[1])

    frames = []
    for im_id, im_data in im_id_to_image.items():
        rotation = qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)

        w2c = np.concatenate([rotation, translation], axis=1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0)
        c2w = np.linalg.inv(w2c)

        c2w[0:3, 1:3] *= -1
        if not keep_original_world_coordinate:
            c2w = c2w[np.array([0, 2, 1, 3]), :]
            c2w[2, :] *= -1

        frame: Dict[str, Any] = {
            "file_path": f"{image_dir.rstrip('/')}/{im_data.name}",
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_id,
        }

        if not use_single_camera_mode:
            frame.update(parse_colmap_camera_params(cam_id_to_camera[im_data.camera_id]))

        frames.append(frame)

    out["frames"] = frames

    if not keep_original_world_coordinate:
        applied_transform = np.eye(4)[:3, :]
        applied_transform = applied_transform[np.array([0, 2, 1]), :]
        applied_transform[2, :] *= -1
        out["applied_transform"] = applied_transform.tolist()

    return out


@dataclass
class CreateTransforms:
    model_dir: Path = Path(".")
    output_file: Path = Path(".")
    image_dir: str = "./images"
    keep_original_world_coordinate: bool = False
    use_single_camera_mode: bool = True

    def main(self) -> None:
        output_file = self.output_file
        if output_file == Path(".") or output_file.is_dir():
            output_file = output_file / "transforms.json"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        transforms = create_transforms_data(
            model_dir=self.model_dir,
            image_dir=self.image_dir,
            keep_original_world_coordinate=self.keep_original_world_coordinate,
            use_single_camera_mode=self.use_single_camera_mode,
        )
        output_file.write_text(json.dumps(transforms, indent=4), encoding="utf-8")
        CONSOLE.print(f"[bold green]:white_check_mark: Saved transforms to {output_file}")


def entrypoint() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", "--model-dir", default=".", help="COLMAP model directory containing cameras.bin and images.bin")
    parser.add_argument("--output_file", "--output-file", default=".", help="Output transforms.json file or directory")
    parser.add_argument("--image_dir", "--image-dir", default="./images", help="Prefix used for frame file paths")
    parser.add_argument(
        "--keep_original_world_coordinate",
        "--keep-original-world-coordinate",
        action="store_true",
        help="Keep COLMAP world coordinates instead of applying nerfstudio z-up transform",
    )
    parser.add_argument(
        "--use_single_camera_mode",
        "--use-single-camera-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write shared camera intrinsics once when possible",
    )
    args = parser.parse_args()
    CreateTransforms(
        model_dir=Path(args.model_dir),
        output_file=Path(args.output_file),
        image_dir=args.image_dir,
        keep_original_world_coordinate=args.keep_original_world_coordinate,
        use_single_camera_mode=args.use_single_camera_mode,
    ).main()


if __name__ == "__main__":
    entrypoint()
