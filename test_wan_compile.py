# SPDX-License-Identifier: Apache-2.0
import os

import torch

from fastvideo.v1.configs.models.dits import WanVideoConfig
from fastvideo.v1.configs.pipelines import PipelineConfig
from fastvideo.v1.distributed import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel)
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.models.loader.component_loader import TransformerLoader
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.utils import maybe_download_model

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"

BASE_MODEL_PATH = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_PATH = maybe_download_model(BASE_MODEL_PATH,
                                  local_dir=os.path.join(
                                      'data', BASE_MODEL_PATH))
TRANSFORMER_PATH = os.path.join(MODEL_PATH, "transformer")

maybe_init_distributed_environment_and_model_parallel(1, 1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
precision = torch.bfloat16
precision_str = "bf16"
args = FastVideoArgs(model_path=TRANSFORMER_PATH,
                     use_cpu_offload=True,
                     pipeline_config=PipelineConfig(
                         dit_config=WanVideoConfig(),
                         dit_precision=precision_str))
args.device = device

loader = TransformerLoader()
model = loader.load(TRANSFORMER_PATH, args).to(dtype=precision)

# Set both models to eval mode
model = model.eval()
# model = torch.compile(model)
# Create identical inputs for both models
batch_size = 1
seq_len = 30

# Video latents [B, C, T, H, W]
hidden_states = torch.randn(batch_size,
                            16,
                            21,
                            160,
                            90,
                            device=device,
                            dtype=precision)

# Text embeddings [B, L, D] (including global token)
encoder_hidden_states = torch.randn(batch_size,
                                    seq_len + 1,
                                    4096,
                                    device=device,
                                    dtype=precision)

# Timestep
timestep = torch.tensor([500], device=device, dtype=precision)

forward_batch = ForwardBatch(data_type="dummy", )

with torch.amp.autocast('cuda', dtype=precision), set_forward_context(
        current_timestep=0,
        attn_metadata=None,
        forward_batch=forward_batch,
):
    output1 = model(hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep)
    torch.compiler.set_stance("force_eager")
    output2 = model(hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep)
    torch.compiler.set_stance("default")

# Check if outputs have the same shape
assert output1.shape == output2.shape, f"Output shapes don't match: {output1.shape} vs {output2.shape}"
assert output1.dtype == output2.dtype, f"Output dtype don't match: {output1.dtype} vs {output2.dtype}"

# Check if outputs are similar (allowing for small numerical differences)
max_diff = torch.max(torch.abs(output1 - output2))
mean_diff = torch.mean(torch.abs(output1 - output2))
logger.info("Max Diff: %s", max_diff.item())
logger.info("Mean Diff: %s", mean_diff.item())
# assert max_diff < 1e-3, f"Maximum difference between outputs: {max_diff.item()}"
# # mean diff
# assert mean_diff < 1e-3, f"Mean difference between outputs: {mean_diff.item()}"

cleanup_dist_env_and_memory()
