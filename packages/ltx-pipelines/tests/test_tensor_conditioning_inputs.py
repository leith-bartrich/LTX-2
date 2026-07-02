"""Tests for pre-decoded tensor conditioning inputs (path-or-tensor pass-through)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ltx_pipelines.iclora_utils import append_ic_lora_reference_video_conditionings
from ltx_pipelines.utils.media_io import (
    frames_to_unit_range,
    load_image_and_preprocess,
    load_video_conditioning_hdr,
    preprocess_tensor_frames,
)

DEVICE = torch.device("cpu")
DTYPE = torch.float32


def test_frames_to_unit_range_uint8() -> None:
    frames = torch.tensor([0, 128, 255], dtype=torch.uint8)
    result = frames_to_unit_range(frames)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([0.0, 128 / 255, 1.0]))


def test_frames_to_unit_range_float_passthrough() -> None:
    frames = torch.tensor([0.0, 0.5004, 1.0], dtype=torch.float64)
    result = frames_to_unit_range(frames)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor([0.0, 0.5004, 1.0]))


def test_preprocess_tensor_frames_image_shape_and_range() -> None:
    image = torch.rand(48, 64, 3)
    result = preprocess_tensor_frames(image, height=32, width=32, dtype=DTYPE, device=DEVICE)
    assert result.shape == (1, 3, 1, 32, 32)
    assert result.min() >= -1.0
    assert result.max() <= 1.0


def test_preprocess_tensor_frames_video_shape() -> None:
    video = torch.randint(0, 256, (5, 48, 64, 3), dtype=torch.uint8)
    result = preprocess_tensor_frames(video, height=32, width=32, dtype=torch.bfloat16, device=DEVICE)
    assert result.shape == (1, 3, 5, 32, 32)
    assert result.dtype == torch.bfloat16


def test_preprocess_tensor_frames_preserves_precision_beyond_8bit() -> None:
    # A float value that is not representable on the uint8 grid must survive.
    value = 0.5004
    image = torch.full((32, 32, 3), value)
    result = preprocess_tensor_frames(image, height=32, width=32, dtype=DTYPE, device=DEVICE)
    assert torch.allclose(result, torch.full_like(result, value * 2.0 - 1.0), atol=1e-6)


def test_load_image_and_preprocess_tensor_matches_path(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    image_path = tmp_path / "image.png"
    Image.fromarray(pixels).save(image_path)

    from_path = load_image_and_preprocess(str(image_path), height=32, width=32, dtype=DTYPE, device=DEVICE, crf=0)
    from_tensor = load_image_and_preprocess(torch.from_numpy(pixels), height=32, width=32, dtype=DTYPE, device=DEVICE)
    assert from_path.shape == from_tensor.shape
    assert torch.allclose(from_path, from_tensor, atol=1e-5)


def test_load_image_and_preprocess_rejects_bad_tensor_shape() -> None:
    with pytest.raises(ValueError, match=r"\(H, W, C\)"):
        load_image_and_preprocess(torch.rand(2, 48, 64, 3), height=32, width=32, dtype=DTYPE, device=DEVICE)


def test_load_video_conditioning_hdr_tensor_input() -> None:
    video = torch.rand(4, 48, 64, 3)
    frames = list(load_video_conditioning_hdr(video, height=32, width=32, frame_cap=3, dtype=DTYPE, device=DEVICE))
    assert len(frames) == 3
    for frame in frames:
        assert frame.shape == (1, 3, 1, 32, 32)
        assert torch.isfinite(frame).all()


def test_load_video_conditioning_hdr_rejects_bad_tensor_shape() -> None:
    frames = load_video_conditioning_hdr(
        torch.rand(48, 64, 3), height=32, width=32, frame_cap=1, dtype=DTYPE, device=DEVICE
    )
    with pytest.raises(ValueError, match=r"\(F, H, W, C\)"):
        next(iter(frames))


class _IdentityEncoder:
    """Stands in for VideoEncoder: returns its input as the 'latent'."""

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return video


def test_append_ic_lora_reference_video_conditionings_tensor_input() -> None:
    video = torch.rand(9, 48, 64, 3)
    conditionings: list = []
    append_ic_lora_reference_video_conditionings(
        conditionings,
        [(video, 0.7)],
        height=32,
        width=32,
        num_frames=5,
        video_encoder=_IdentityEncoder(),
        dtype=DTYPE,
        device=DEVICE,
        reference_downscale_factor=1,
        conditioning_attention_strength=1.0,
        conditioning_attention_mask=None,
    )
    assert len(conditionings) == 1
    cond = conditionings[0]
    assert cond.strength == 0.7
    # Frames are capped at num_frames before encoding.
    assert cond.latent.shape == (1, 3, 5, 32, 32)


def test_append_ic_lora_reference_video_conditionings_rejects_bad_tensor_shape() -> None:
    with pytest.raises(ValueError, match=r"\(F, H, W, C\)"):
        append_ic_lora_reference_video_conditionings(
            [],
            [(torch.rand(48, 64, 3), 1.0)],
            height=32,
            width=32,
            num_frames=5,
            video_encoder=_IdentityEncoder(),
            dtype=DTYPE,
            device=DEVICE,
            reference_downscale_factor=1,
            conditioning_attention_strength=1.0,
            conditioning_attention_mask=None,
        )
