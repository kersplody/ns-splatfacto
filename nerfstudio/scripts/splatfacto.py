#!/usr/bin/env python
"""Slim CLI for Splatfacto training, eval, and export workflows."""

from __future__ import annotations

import json
import os
import random
import socket
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
import yaml
from typing_extensions import Annotated

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import comms, profiler
from nerfstudio.utils.available_devices import get_available_devices
from nerfstudio.utils.rich_utils import CONSOLE

DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


def _find_free_port() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_splatfacto_config(variant: Literal["splatfacto", "splatfacto-big"] = "splatfacto") -> TrainerConfig:
    model_kwargs: Dict[str, Any] = {}
    if variant == "splatfacto-big":
        model_kwargs.update(
            {
                "cull_alpha_thresh": 0.005,
                "densify_grad_thresh": 0.0005,
            }
        )

    return TrainerConfig(
        method_name=variant,
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=SplatfactoModelConfig(**model_kwargs),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="none",
    )


def train_loop(local_rank: int, world_size: int, config: TrainerConfig, global_rank: int = 0):
    _set_random_seed(config.machine.seed + global_rank)
    trainer = config.setup(local_rank=local_rank, world_size=world_size)
    trainer.setup()
    trainer.train()


def _distributed_worker(
    local_rank: int,
    main_func: Callable,
    world_size: int,
    num_devices_per_machine: int,
    machine_rank: int,
    dist_url: str,
    config: TrainerConfig,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> Any:
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_devices_per_machine + local_rank

    dist.init_process_group(
        backend="nccl" if device_type == "cuda" else "gloo",
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank,
        timeout=timeout,
    )
    assert comms.LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_devices_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_devices_per_machine, (i + 1) * num_devices_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comms.LOCAL_PROCESS_GROUP = pg

    assert num_devices_per_machine <= torch.cuda.device_count()
    output = main_func(local_rank, world_size, config, global_rank)
    comms.synchronize()
    dist.destroy_process_group()
    return output


def launch(
    main_func: Callable,
    num_devices_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str = "auto",
    config: Optional[TrainerConfig] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
    device_type: Literal["cpu", "cuda", "mps"] = "cuda",
) -> None:
    assert config is not None
    world_size = num_machines * num_devices_per_machine
    if world_size == 0:
        raise ValueError("world_size cannot be 0")
    if world_size == 1:
        try:
            main_func(local_rank=0, world_size=world_size, config=config)
        finally:
            profiler.flush_profiler(config.logging)
        return

    if dist_url == "auto":
        assert num_machines == 1, "dist_url=auto is not supported for multi-machine jobs."
        dist_url = f"tcp://127.0.0.1:{_find_free_port()}"

    process_context = mp.spawn(
        _distributed_worker,
        nprocs=num_devices_per_machine,
        join=False,
        args=(main_func, world_size, num_devices_per_machine, machine_rank, dist_url, config, timeout, device_type),
    )
    assert process_context is not None
    try:
        process_context.join()
    finally:
        profiler.flush_profiler(config.logging)


def run_train(config: TrainerConfig) -> None:
    available_device_types = get_available_devices()
    if config.machine.device_type not in available_device_types:
        raise RuntimeError(
            f"Specified device type '{config.machine.device_type}' is not available. "
            f"Available device types: {available_device_types}."
        )

    if config.data:
        config.pipeline.datamanager.data = config.data

    if config.load_config:
        config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)

    config.set_timestamp()
    config.print_to_terminal()
    config.save_config()

    launch(
        main_func=train_loop,
        num_devices_per_machine=config.machine.num_devices,
        device_type=config.machine.device_type,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )


def _prepare_train_config(config: TrainerConfig) -> TrainerConfig:
    available_device_types = get_available_devices()
    if config.machine.device_type not in available_device_types:
        raise RuntimeError(
            f"Specified device type '{config.machine.device_type}' is not available. "
            f"Available device types: {available_device_types}."
        )

    if config.data:
        config.pipeline.datamanager.data = config.data

    if config.load_config:
        config = yaml.load(config.load_config.read_text(), Loader=yaml.Loader)

    config.set_timestamp()
    config.print_to_terminal()
    config.save_config()
    return config


def run_train_single_process(config: TrainerConfig) -> Any:
    config = _prepare_train_config(config)
    _set_random_seed(config.machine.seed)
    trainer = config.setup(local_rank=0, world_size=1)
    trainer.setup()
    try:
        trainer.train()
    finally:
        profiler.flush_profiler(config.logging)
    return trainer


def eval_setup_splatfacto(
    config_path: Path,
    test_mode: Literal["test", "val", "inference"] = "test",
) -> Tuple[TrainerConfig, Pipeline, Path, int]:
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    config.pipeline.datamanager._target = FullImageDatamanager
    config.load_dir = config.get_checkpoint_dir()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    assert config.load_dir is not None
    load_step = config.load_step
    if load_step is None:
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    loaded_state = torch.load(load_path, map_location="cpu", weights_only=False)
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")
    return config, pipeline, load_path, load_step


def collect_camera_poses_for_dataset(
    dataset: Optional[InputDataset], camera_optimizer: Optional[CameraOptimizer] = None
) -> List[Dict[str, Any]]:
    if dataset is None:
        return []

    cameras = dataset.cameras
    image_filenames = dataset.image_filenames
    frames: List[Dict[str, Any]] = []

    for idx in range(len(cameras)):
        image_filename = image_filenames[idx]
        if camera_optimizer is None:
            transform = cameras.camera_to_worlds[idx].tolist()
        else:
            camera = cameras[idx : idx + 1]
            assert camera.metadata is not None
            camera.metadata["cam_idx"] = idx
            transform = camera_optimizer.apply_to_camera(camera).tolist()[0]

        frames.append(
            {
                "file_path": str(image_filename),
                "transform": transform,
            }
        )

    return frames


def collect_camera_poses(pipeline: VanillaPipeline) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_dataset = pipeline.datamanager.train_dataset
    assert isinstance(train_dataset, InputDataset)

    eval_dataset = pipeline.datamanager.eval_dataset
    assert isinstance(eval_dataset, InputDataset)

    camera_optimizer = None
    if hasattr(pipeline.model, "camera_optimizer"):
        camera_optimizer = pipeline.model.camera_optimizer
        assert isinstance(camera_optimizer, CameraOptimizer)

    return (
        collect_camera_poses_for_dataset(train_dataset, camera_optimizer),
        collect_camera_poses_for_dataset(eval_dataset),
    )


def write_eval_results(
    config: TrainerConfig,
    checkpoint_path: Path,
    pipeline: Pipeline,
    output_path: Path,
    render_output_path: Optional[Path] = None,
) -> None:
    assert output_path.suffix == ".json"
    if render_output_path is not None:
        render_output_path.mkdir(parents=True, exist_ok=True)
    metrics_dict = pipeline.get_average_eval_image_metrics(output_path=render_output_path, get_std=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_info = {
        "experiment_name": config.experiment_name,
        "method_name": config.method_name,
        "checkpoint": str(checkpoint_path),
        "results": metrics_dict,
    }
    output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
    CONSOLE.print(f"Saved results to: {output_path}")


def export_gaussian_splat_from_pipeline(
    pipeline: Pipeline,
    output_dir: Path,
    output_filename: str = "splat.ply",
    obb_center: Optional[Tuple[float, float, float]] = None,
    obb_rotation: Optional[Tuple[float, float, float]] = None,
    obb_scale: Optional[Tuple[float, float, float]] = None,
    ply_color_mode: Literal["sh_coeffs", "rgb"] = "sh_coeffs",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    assert isinstance(pipeline.model, SplatfactoModel)
    model = cast(SplatfactoModel, pipeline.model)

    filename = output_dir / output_filename
    map_to_tensors: OrderedDict[str, np.ndarray] = OrderedDict()

    with torch.no_grad():
        positions = model.means.cpu().numpy()
        count = positions.shape[0]
        n = count
        map_to_tensors["x"] = positions[:, 0]
        map_to_tensors["y"] = positions[:, 1]
        map_to_tensors["z"] = positions[:, 2]
        map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

        if ply_color_mode == "rgb":
            colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
            colors = (colors * 255).astype(np.uint8)
            map_to_tensors["red"] = colors[:, 0]
            map_to_tensors["green"] = colors[:, 1]
            map_to_tensors["blue"] = colors[:, 2]
        else:
            shs_0 = model.shs_0.contiguous().cpu().numpy()
            for i in range(shs_0.shape[1]):
                map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

        if model.config.sh_degree > 0 and ply_color_mode == "sh_coeffs":
            shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
            shs_rest = shs_rest.reshape((n, -1))
            for i in range(shs_rest.shape[-1]):
                map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]

        map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()

        scales = model.scales.data.cpu().numpy()
        for i in range(3):
            map_to_tensors[f"scale_{i}"] = scales[:, i, None]

        quats = model.quats.data.cpu().numpy()
        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i, None]

        if obb_center is not None and obb_rotation is not None and obb_scale is not None:
            crop_obb = OrientedBox.from_params(obb_center, obb_rotation, obb_scale)
            mask = crop_obb.within(torch.from_numpy(positions)).numpy()
            for key in list(map_to_tensors.keys()):
                map_to_tensors[key] = map_to_tensors[key][mask]
            count = map_to_tensors["x"].shape[0]
            n = count

    select = np.ones(n, dtype=bool)
    for tensor in map_to_tensors.values():
        select = np.logical_and(select, np.isfinite(tensor).all(axis=-1))

    low_opacity_gaussians = map_to_tensors["opacity"].squeeze(axis=-1) < -5.5373
    select[low_opacity_gaussians] = 0

    if np.sum(select) < n:
        for key in list(map_to_tensors.keys()):
            map_to_tensors[key] = map_to_tensors[key][select]
        count = int(np.sum(select))

    ExportGaussianSplat.write_ply(str(filename), count, map_to_tensors)
    CONSOLE.print(f"[bold green]:white_check_mark: Saved splat PLY to {filename}")


def _resolve_train_eval_export_outputs(
    output_dir: Optional[Path],
    run_name: Optional[str],
    eval_output_path: Path,
    export_output_dir: Path,
    output_filename: str,
) -> Tuple[Path, Path]:
    if output_dir is None and run_name is None:
        return eval_output_path, export_output_dir / output_filename

    if output_dir is None or run_name is None:
        raise ValueError("--output-dir and --run-name must be provided together.")

    return output_dir / f"{run_name}.eval.json", output_dir / f"{run_name}.ply"


SplatfactoTrainConfig = tyro.extras.subcommand_type_from_defaults(
    {
        "splatfacto": build_splatfacto_config("splatfacto"),
        "splatfacto-big": build_splatfacto_config("splatfacto-big"),
    },
    prefix_names=False,
)


@dataclass
class TrainSplatfacto:
    config: SplatfactoTrainConfig = field(default_factory=lambda: build_splatfacto_config("splatfacto"))

    def main(self) -> None:
        config = cast(TrainerConfig, self.config)
        config.vis = "none"
        run_train(config)


@dataclass
class TrainEvalExportSplatfacto:
    config: SplatfactoTrainConfig = field(default_factory=lambda: build_splatfacto_config("splatfacto"))
    output_dir: Optional[Path] = None
    run_name: Optional[str] = None
    eval_output_path: Path = Path("eval.json")
    render_output_path: Optional[Path] = None
    export_output_dir: Path = Path("exports/splat")
    output_filename: str = "splat.ply"
    obb_center: Optional[Tuple[float, float, float]] = None
    obb_rotation: Optional[Tuple[float, float, float]] = None
    obb_scale: Optional[Tuple[float, float, float]] = None
    ply_color_mode: Literal["sh_coeffs", "rgb"] = "sh_coeffs"

    def main(self) -> None:
        config = cast(TrainerConfig, self.config)
        config.vis = "none"
        if config.machine.num_devices != 1 or config.machine.num_machines != 1:
            raise ValueError("train-eval-export only supports a single-process, one-GPU run.")

        trainer = run_train_single_process(config)
        checkpoint_path = trainer.checkpoint_dir / f"step-{trainer.step:09d}.ckpt"
        pipeline = trainer.pipeline
        eval_output_path, export_output_path = _resolve_train_eval_export_outputs(
            self.output_dir,
            self.run_name,
            self.eval_output_path,
            self.export_output_dir,
            self.output_filename,
        )

        write_eval_results(
            trainer.config,
            checkpoint_path,
            pipeline,
            output_path=eval_output_path,
            render_output_path=self.render_output_path,
        )
        export_gaussian_splat_from_pipeline(
            pipeline,
            output_dir=export_output_path.parent,
            output_filename=export_output_path.name,
            obb_center=self.obb_center,
            obb_rotation=self.obb_rotation,
            obb_scale=self.obb_scale,
            ply_color_mode=self.ply_color_mode,
        )


@dataclass
class EvalSplatfacto:
    load_config: Path
    output_path: Path = Path("output.json")
    render_output_path: Optional[Path] = None

    def main(self) -> None:
        config, pipeline, checkpoint_path, _ = eval_setup_splatfacto(self.load_config)
        write_eval_results(config, checkpoint_path, pipeline, self.output_path, self.render_output_path)


@dataclass
class ExportCameraPoses:
    load_config: Path
    output_dir: Path

    def main(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _, pipeline, _, _ = eval_setup_splatfacto(self.load_config)
        assert isinstance(pipeline, VanillaPipeline)
        train_frames, eval_frames = collect_camera_poses(pipeline)

        for file_name, frames in [("transforms_train.json", train_frames), ("transforms_eval.json", eval_frames)]:
            if len(frames) == 0:
                continue
            output_file_path = self.output_dir / file_name
            output_file_path.write_text(json.dumps(frames, indent=4), "utf8")
            CONSOLE.print(f"[bold green]:white_check_mark: Saved poses to {output_file_path}")


@dataclass
class ExportGaussianSplat:
    load_config: Path
    output_dir: Path
    output_filename: str = "splat.ply"
    obb_center: Optional[Tuple[float, float, float]] = None
    obb_rotation: Optional[Tuple[float, float, float]] = None
    obb_scale: Optional[Tuple[float, float, float]] = None
    ply_color_mode: Literal["sh_coeffs", "rgb"] = "sh_coeffs"

    @staticmethod
    def write_ply(filename: str, count: int, map_to_tensors: OrderedDict[str, np.ndarray]) -> None:
        if not all(tensor.size == count for tensor in map_to_tensors.values()):
            raise ValueError("Count does not match the length of all tensors")

        with open(filename, "wb") as ply_file:
            nerfstudio_version = version("ns-splatfacto")
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")
            ply_file.write(f"comment Generated by Nerfstudio {nerfstudio_version}\n".encode())
            ply_file.write(b"comment Vertical Axis: z\n")
            ply_file.write(f"element vertex {count}\n".encode())

            for key, tensor in map_to_tensors.items():
                data_type = "float" if tensor.dtype.kind == "f" else "uchar"
                ply_file.write(f"property {data_type} {key}\n".encode())

            ply_file.write(b"end_header\n")

            for i in range(count):
                for tensor in map_to_tensors.values():
                    value = tensor[i]
                    if tensor.dtype.kind == "f":
                        ply_file.write(np.float32(value).tobytes())
                    else:
                        ply_file.write(value.tobytes())

    def main(self) -> None:
        _, pipeline, _, _ = eval_setup_splatfacto(self.load_config, test_mode="inference")
        export_gaussian_splat_from_pipeline(
            pipeline,
            output_dir=self.output_dir,
            output_filename=self.output_filename,
            obb_center=self.obb_center,
            obb_rotation=self.obb_rotation,
            obb_scale=self.obb_scale,
            ply_color_mode=self.ply_color_mode,
        )


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[TrainSplatfacto, tyro.conf.subcommand(name="train")],
        Annotated[TrainEvalExportSplatfacto, tyro.conf.subcommand(name="train-eval-export")],
        Annotated[EvalSplatfacto, tyro.conf.subcommand(name="eval")],
        Annotated[ExportGaussianSplat, tyro.conf.subcommand(name="export-gaussian-splat")],
        Annotated[ExportCameraPoses, tyro.conf.subcommand(name="export-camera-poses")],
    ]
]


def entrypoint() -> None:
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()
