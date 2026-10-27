#!/usr/bin/env python3
"""
YOLO26 TOP/BOTTOM classification training with full-garment-preserving letterbox preprocessing.

Target environment:
- Python 3.10
- Ultralytics 8.4.24
- PyTorch 2.6.0
- torchvision 0.21.0

Key behavior:
- Replaces train RandomResizedCrop/RandAugment/RandomErasing with:
  gray letterbox -> optional horizontal flip -> mild ColorJitter -> tensor
- Replaces val Resize/CenterCrop with:
  gray letterbox -> tensor
- Uses Ultralytics' current classification normalization defaults:
  mean=(0, 0, 0), std=(1, 1, 1), i.e. ToTensor's [0, 1] range is retained.
- Applies fraction sampling independently per class instead of slicing the
  alphabetically sorted ImageFolder sample list.
- Exposes the same build_letterbox_transform() for future inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import traceback
from collections import Counter, defaultdict
from copy import copy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import ultralytics
from ultralytics import YOLO
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator
from ultralytics.utils import LOGGER, colorstr


DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)
SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


@dataclass(frozen=True)
class TransformConfig:
    """Configuration shared by training, validation, preview, and inference."""

    imgsz: int = 224
    gray_value: int = 114
    hflip: float = 0.5
    brightness: float = 0.10
    contrast: float = 0.10
    saturation: float = 0.10
    seed: int = 42
    mean: tuple[float, float, float] = DEFAULT_MEAN
    std: tuple[float, float, float] = DEFAULT_STD


class LetterboxSquare:
    """Resize while preserving aspect ratio and center on a square RGB canvas."""

    def __init__(
        self,
        size: int,
        fill: int | tuple[int, int, int] = 114,
        interpolation: Image.Resampling = Image.Resampling.BILINEAR,
    ) -> None:
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"size must be a positive integer, got {size!r}")

        if isinstance(fill, int):
            if not 0 <= fill <= 255:
                raise ValueError(f"fill must be in [0, 255], got {fill}")
            fill_tuple = (fill, fill, fill)
        else:
            if len(fill) != 3 or any(not 0 <= int(v) <= 255 for v in fill):
                raise ValueError(f"fill must contain three values in [0, 255], got {fill!r}")
            fill_tuple = tuple(int(v) for v in fill)

        self.size = size
        self.fill = fill_tuple
        self.interpolation = interpolation

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {(width, height)}")

        scale = min(self.size / width, self.size / height)
        new_width = max(1, min(self.size, int(round(width * scale))))
        new_height = max(1, min(self.size, int(round(height * scale))))

        resized = image.resize((new_width, new_height), self.interpolation)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)

        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2
        canvas.paste(resized, (left, top))
        return canvas

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(size={self.size}, fill={self.fill}, "
            f"interpolation={self.interpolation.name})"
        )


def build_letterbox_transform(
    imgsz: int = 224,
    gray_value: int = 114,
    train: bool = False,
    hflip: float = 0.5,
    brightness: float = 0.10,
    contrast: float = 0.10,
    saturation: float = 0.10,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
) -> T.Compose:
    """
    Build the transform reused by train, val, preview, and future inference.

    No crop, rotation, vertical flip, auto augmentation, cutout, or random
    erasing is included.
    """
    transforms: list[Any] = [
        LetterboxSquare(size=imgsz, fill=(gray_value, gray_value, gray_value))
    ]

    if train:
        if hflip > 0.0:
            transforms.append(T.RandomHorizontalFlip(p=hflip))

        if brightness > 0.0 or contrast > 0.0 or saturation > 0.0:
            transforms.append(
                T.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=0.0,
                )
            )

    transforms.extend(
        [
            T.ToTensor(),
            T.Normalize(
                mean=torch.tensor(mean, dtype=torch.float32),
                std=torch.tensor(std, dtype=torch.float32),
            ),
        ]
    )
    return T.Compose(transforms)


def preprocess_pil_for_inference(
    image: Image.Image,
    imgsz: int = 224,
    gray_value: int = 114,
) -> torch.Tensor:
    """
    Convert a PIL RGB image to one classifier input tensor of shape [1, 3, H, W].

    This deliberately uses the same validation transform as training.
    """
    transform = build_letterbox_transform(
        imgsz=imgsz,
        gray_value=gray_value,
        train=False,
    )
    return transform(image.convert("RGB")).unsqueeze(0)


def _stratified_fraction(
    samples: list[tuple[str, int]],
    fraction: float,
    seed: int,
) -> list[tuple[str, int]]:
    """Select the requested fraction independently inside every class."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    if fraction >= 1.0:
        return list(samples)

    grouped: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for sample in samples:
        grouped[int(sample[1])].append(sample)

    rng = random.Random(seed)
    selected: list[tuple[str, int]] = []

    for class_index in sorted(grouped):
        class_samples = list(grouped[class_index])
        rng.shuffle(class_samples)
        keep = max(1, int(round(len(class_samples) * fraction)))
        selected.extend(class_samples[:keep])

    # Stable order keeps verification/cache behavior reproducible.
    selected.sort(key=lambda item: item[0])
    return selected


class LetterboxClassificationDataset(ClassificationDataset):
    """Ultralytics ClassificationDataset with custom transforms and fair fraction sampling."""

    def __init__(
        self,
        root: str,
        args: Any,
        augment: bool = False,
        prefix: str = "",
        transform_config: TransformConfig | None = None,
    ) -> None:
        config = transform_config or TransformConfig(
            imgsz=int(args.imgsz),
            seed=int(getattr(args, "seed", 42)),
        )

        # Reproduce Ultralytics 8.4.24 ClassificationDataset initialization,
        # but apply fraction per class before verify_images().
        try:
            self.base = ImageFolder(root=root, allow_empty=True)
        except TypeError:
            self.base = ImageFolder(root=root)

        self.samples = list(self.base.samples)
        self.root = self.base.root
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""

        requested_fraction = float(getattr(args, "fraction", 1.0))
        if augment and requested_fraction < 1.0:
            before_counts = Counter(int(sample[1]) for sample in self.samples)
            self.samples = _stratified_fraction(
                self.samples,
                fraction=requested_fraction,
                seed=config.seed,
            )
            after_counts = Counter(int(sample[1]) for sample in self.samples)
            LOGGER.info(
                "%sstratified fraction=%.6f: before=%s after=%s",
                self.prefix,
                requested_fraction,
                dict(sorted(before_counts.items())),
                dict(sorted(after_counts.items())),
            )

        cache_value = getattr(args, "cache", False)
        self.cache_ram = cache_value is True or str(cache_value).lower() == "ram"
        if self.cache_ram:
            LOGGER.warning(
                "Classification cache_ram has a known memory leak in the target "
                "Ultralytics release; forcing cache_ram=False."
            )
            self.cache_ram = False

        self.cache_disk = str(cache_value).lower() == "disk"
        self.samples = self.verify_images()
        self.samples = [
            [*list(sample), Path(sample[0]).with_suffix(".npy"), None]
            for sample in self.samples
        ]

        self.torch_transforms = build_letterbox_transform(
            imgsz=config.imgsz,
            gray_value=config.gray_value,
            train=augment,
            hflip=config.hflip,
            brightness=config.brightness,
            contrast=config.contrast,
            saturation=config.saturation,
            mean=config.mean,
            std=config.std,
        )

        verified_counts = Counter(int(sample[1]) for sample in self.samples)
        LOGGER.info(
            "%sletterbox transform=%s",
            self.prefix,
            self.torch_transforms,
        )
        LOGGER.info(
            "%sverified class counts=%s",
            self.prefix,
            dict(sorted(verified_counts.items())),
        )


class LetterboxClassificationValidator(ClassificationValidator):
    """Validator that builds the same no-crop letterbox validation dataset."""

    transform_config = TransformConfig()

    def build_dataset(self, img_path: str) -> LetterboxClassificationDataset:
        return LetterboxClassificationDataset(
            root=img_path,
            args=self.args,
            augment=False,
            prefix=self.args.split,
            transform_config=self.transform_config,
        )


class LetterboxClassificationTrainer(ClassificationTrainer):
    """Trainer that uses letterbox datasets for both train and validation."""

    transform_config = TransformConfig()

    def build_dataset(
        self,
        img_path: str,
        mode: str = "train",
        batch: Any = None,
    ) -> LetterboxClassificationDataset:
        del batch  # unused by Ultralytics classification dataset
        return LetterboxClassificationDataset(
            root=img_path,
            args=self.args,
            augment=mode == "train",
            prefix=mode,
            transform_config=self.transform_config,
        )

    def get_validator(self) -> LetterboxClassificationValidator:
        self.loss_names = ["loss"]
        LetterboxClassificationValidator.transform_config = self.transform_config
        return LetterboxClassificationValidator(
            self.test_loader,
            self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )


def _count_folder_images(split_dir: Path) -> tuple[list[str], dict[str, int]]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")

    class_names = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    if not class_names:
        raise RuntimeError(f"No class directories found under: {split_dir}")

    counts: dict[str, int] = {}
    for class_name in class_names:
        class_dir = split_dir / class_name
        counts[class_name] = sum(
            1
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
    return class_names, counts


def inspect_dataset(data_root: Path) -> dict[str, Any]:
    """Print and return train/val class order and image counts."""
    report: dict[str, Any] = {"data_root": str(data_root)}

    expected_classes: list[str] | None = None
    for split in ("train", "val"):
        class_names, counts = _count_folder_images(data_root / split)
        mapping = {name: index for index, name in enumerate(class_names)}

        if expected_classes is None:
            expected_classes = class_names
        elif class_names != expected_classes:
            raise RuntimeError(
                f"Class directory mismatch: train={expected_classes}, {split}={class_names}"
            )

        report[split] = {
            "classes": class_names,
            "class_to_idx": mapping,
            "counts": counts,
            "total": sum(counts.values()),
        }

    print("\n" + "=" * 80)
    print("DATASET")
    print("=" * 80)
    print(f"root: {data_root}")
    for split in ("train", "val"):
        print(
            f"{split}: total={report[split]['total']:,}, "
            f"class_to_idx={report[split]['class_to_idx']}, "
            f"counts={report[split]['counts']}"
        )

    train_mapping = report["train"]["class_to_idx"]
    if train_mapping != {"BOTTOM": 0, "TOP": 1}:
        print(
            "[WARNING] Expected alphabetical mapping {'BOTTOM': 0, 'TOP': 1}, "
            f"but found {train_mapping}."
        )
    else:
        print("class order confirmed: BOTTOM=0, TOP=1")

    return report


def _balanced_preview_samples(
    split_dir: Path,
    count: int,
    seed: int,
) -> tuple[ImageFolder, list[tuple[str, int]]]:
    try:
        dataset = ImageFolder(root=str(split_dir), allow_empty=True)
    except TypeError:
        dataset = ImageFolder(root=str(split_dir))

    grouped: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for sample in dataset.samples:
        grouped[int(sample[1])].append(sample)

    rng = random.Random(seed)
    per_class = max(1, math.ceil(count / max(1, len(grouped))))
    selected: list[tuple[str, int]] = []

    for class_index in sorted(grouped):
        pool = list(grouped[class_index])
        rng.shuffle(pool)
        selected.extend(pool[:per_class])

    rng.shuffle(selected)
    return dataset, selected[:count]


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().float().clamp(0.0, 1.0)
    array = (
        tensor.permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def save_preview_sheet(
    split_dir: Path,
    output_path: Path,
    config: TransformConfig,
    train: bool,
    count: int,
) -> None:
    """Save a balanced contact sheet produced by the exact custom transform."""
    if count <= 0:
        return

    dataset, selected = _balanced_preview_samples(
        split_dir=split_dir,
        count=count,
        seed=config.seed + (1 if train else 2),
    )

    # Keep random train augmentations reproducible for the contact sheet.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    transform = build_letterbox_transform(
        imgsz=config.imgsz,
        gray_value=config.gray_value,
        train=train,
        hflip=config.hflip,
        brightness=config.brightness,
        contrast=config.contrast,
        saturation=config.saturation,
        mean=config.mean,
        std=config.std,
    )

    tile_width = config.imgsz
    caption_height = 34
    columns = min(6, max(1, math.ceil(math.sqrt(len(selected)))))
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows * (config.imgsz + caption_height)),
        (config.gray_value, config.gray_value, config.gray_value),
    )
    draw = ImageDraw.Draw(sheet)

    for index, (file_path, class_index) in enumerate(selected):
        with Image.open(file_path) as image:
            original_size = image.size
            transformed = _tensor_to_pil(transform(image.convert("RGB")))

        column = index % columns
        row = index // columns
        x = column * tile_width
        y = row * (config.imgsz + caption_height)
        sheet.paste(transformed, (x, y))

        class_name = dataset.classes[class_index]
        caption = f"{class_name}({class_index}) {original_size[0]}x{original_size[1]}"
        draw.rectangle(
            (x, y + config.imgsz, x + tile_width, y + config.imgsz + caption_height),
            fill=(235, 235, 235),
        )
        draw.text(
            (x + 4, y + config.imgsz + 8),
            caption,
            fill=(0, 0, 0),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)
    print(f"preview saved: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train YOLO26 TOP/BOTTOM classifier using aspect-ratio-preserving "
            "gray letterbox transforms for train and validation."
        )
    )

    parser.add_argument("--model", default="yolo26n-cls.pt")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "/home/kusw/fold_project/project_train/data/kfashion_top_bottom_cls"
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("/home/kusw/fold_project/project_train/runs/classify"),
    )
    parser.add_argument("--name", default="kfashion_top_bottom_letterbox")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gray-value", type=int, default=114)
    parser.add_argument("--hflip", type=float, default=0.5)
    parser.add_argument("--brightness", type=float, default=0.10)
    parser.add_argument("--contrast", type=float, default=0.10)
    parser.add_argument("--saturation", type=float, default=0.10)
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume the latest run with --resume, or provide a checkpoint path.",
    )
    parser.add_argument(
        "--cache",
        choices=("false", "disk", "ram"),
        default="false",
        help="Use false initially. RAM cache is forcibly disabled for this target release.",
    )
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save balanced train/val transform contact sheets before training.",
    )
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--preview-count", type=int, default=32)
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Create preview sheets and exit without starting training.",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    if args.batch <= 0:
        raise ValueError("--batch must be a positive integer; batch=-1 is intentionally disabled")
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")
    if not 0 <= args.gray_value <= 255:
        raise ValueError("--gray-value must be in [0, 255]")
    if not 0.0 <= args.hflip <= 1.0:
        raise ValueError("--hflip must be in [0, 1]")
    for name in ("brightness", "contrast", "saturation"):
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"--{name} must be non-negative")
    if args.preview_count < 0:
        raise ValueError("--preview-count must be non-negative")


def print_environment(config: TransformConfig, args: argparse.Namespace) -> None:
    print("=" * 80)
    print("ENVIRONMENT")
    print("=" * 80)
    print(f"python      : {sys.version.splitlines()[0]}")
    print(f"torch       : {torch.__version__}")
    print(f"torchvision : {torchvision.__version__}")
    print(f"ultralytics : {ultralytics.__version__}")
    print(f"cuda        : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu         : {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 80)
    print("CUSTOM TRANSFORMS")
    print("=" * 80)
    print(
        "train:",
        build_letterbox_transform(
            imgsz=config.imgsz,
            gray_value=config.gray_value,
            train=True,
            hflip=config.hflip,
            brightness=config.brightness,
            contrast=config.contrast,
            saturation=config.saturation,
            mean=config.mean,
            std=config.std,
        ),
    )
    print(
        "val/inference:",
        build_letterbox_transform(
            imgsz=config.imgsz,
            gray_value=config.gray_value,
            train=False,
            mean=config.mean,
            std=config.std,
        ),
    )
    print("normalization mean/std:", config.mean, config.std)
    print("RandomResizedCrop: disabled")
    print("CenterCrop       : disabled")
    print("rotation         : disabled")
    print("auto augment     : disabled")
    print("random erasing   : disabled")
    print("vertical flip    : disabled")

    print("\n" + "=" * 80)
    print("RUN ARGUMENTS")
    print("=" * 80)
    print(json.dumps(vars(args), indent=2, ensure_ascii=False, default=str))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(args)

    args.data = args.data.expanduser().resolve()
    args.project = args.project.expanduser().resolve()

    config = TransformConfig(
        imgsz=args.imgsz,
        gray_value=args.gray_value,
        hflip=args.hflip,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        seed=args.seed,
    )

    print_environment(config, args)
    report = inspect_dataset(args.data)

    preview_dir = (
        args.preview_dir.expanduser().resolve()
        if args.preview_dir is not None
        else args.project / f"{args.name}_previews"
    )

    if args.preview or args.preview_only:
        save_preview_sheet(
            split_dir=args.data / "train",
            output_path=preview_dir / "train_letterbox_preview.jpg",
            config=config,
            train=True,
            count=args.preview_count,
        )
        save_preview_sheet(
            split_dir=args.data / "val",
            output_path=preview_dir / "val_letterbox_preview.jpg",
            config=config,
            train=False,
            count=args.preview_count,
        )

        preview_dir.mkdir(parents=True, exist_ok=True)
        with (preview_dir / "preview_config.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "transform": asdict(config),
                    "dataset": report,
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

    if args.preview_only:
        print("preview-only completed; training was not started.")
        return 0

    LetterboxClassificationTrainer.transform_config = config
    LetterboxClassificationValidator.transform_config = config

    cache_value: bool | str
    if args.cache == "false":
        cache_value = False
    else:
        cache_value = args.cache

    train_kwargs: dict[str, Any] = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "fraction": args.fraction,
        "project": str(args.project),
        "name": args.name,
        "seed": args.seed,
        "amp": args.amp,
        "plots": args.plots,
        "val": args.val,
        "cache": cache_value,
        "exist_ok": args.exist_ok,
        # Explicitly neutralize Ultralytics defaults. The custom dataset
        # ignores these geometry/erasing settings, but the run arguments
        # should also record that they are disabled.
        "fliplr": args.hflip,
        "flipud": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "auto_augment": None,
        "erasing": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    }

    if args.resume is not False:
        train_kwargs["resume"] = args.resume

    print("\n" + "=" * 80)
    print("START TRAINING")
    print("=" * 80)
    print("Ultralytics kwargs:")
    print(json.dumps(train_kwargs, indent=2, ensure_ascii=False, default=str))

    model = YOLO(args.model)
    results = model.train(
        trainer=LetterboxClassificationTrainer,
        **train_kwargs,
    )

    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"save_dir: {save_dir}")
    print(f"results: {results}")
    return 0


def _write_exception_log(exc_text: str, project: Path | None, name: str | None) -> Path:
    root = project or Path.cwd()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name or "train_top_bottom_cls_letterbox"
    path = root / f"{safe_name}_error_{timestamp}.txt"
    path.write_text(exc_text, encoding="utf-8")
    return path


if __name__ == "__main__":
    parsed_project: Path | None = None
    parsed_name: str | None = None
    try:
        # Best-effort extraction so an exception log still has a useful location.
        if "--project" in sys.argv:
            parsed_project = Path(sys.argv[sys.argv.index("--project") + 1]).expanduser()
        if "--name" in sys.argv:
            parsed_name = sys.argv[sys.argv.index("--name") + 1]

        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        text = traceback.format_exc()
        error_path = _write_exception_log(text, parsed_project, parsed_name)
        print(text, file=sys.stderr)
        print(f"exception traceback saved: {error_path}", file=sys.stderr)
        raise SystemExit(1)
