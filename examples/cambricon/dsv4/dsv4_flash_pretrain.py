import math
from megatron.bridge.recipes.deepseek import (
    deepseek_v4_flash_pretrain_config,
    deepseek_v4_flash_pretrain_mxfp8_config,
    set_deepseek_v4_pipeline_model_parallel_layout,
)
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.config import ProfilingConfig
if __name__ == "__main__":
    #/workspace/dataset/favorite/soft-data-platform/v1/models/DeepSeek-V4-Flash/
    #/workspace/dataset/favorite/soft-data-platform/v1/models/DeepSeek-V4-Pro/
    hf_path="/workspace/data/models/DeepSeek-V4-Flash"
    cfg = deepseek_v4_flash_pretrain_config(hf_path)
    #cfg = deepseek_v4_flash_pretrain_mxfp8_config()
    cfg.validation.eval_iters = 1
    cfg.checkpoint.save = None
    cfg.checkpoint.load = None
    cfg.logger.log_interval = 1
    cfg.scheduler.lr_warmup_iters = 0
    cfg.logger.tensorboard_dir = None

    # Reduce model scale for testing. When num_layers or mtp_num_layers is
    # changed, csa_compress_ratios must be resized to num_layers +
    # mtp_num_layers and the pipeline layout must be recomputed.
    cfg.model.num_layers = 4  # 43
    cfg.model.mtp_num_layers = 1 #getattr(cfg.model, "mtp_num_layers", 0)

    # Preserve the original CSA compression pattern for the kept base layers
    # and use the original MTP ratio (last entry, typically 0) for MTP layers.
    orig_ratios = cfg.model.csa_compress_ratios
    mtp_ratio = orig_ratios[-1] if orig_ratios else 0
    cfg.model.csa_compress_ratios = (
        orig_ratios[: cfg.model.num_layers] + [mtp_ratio] * cfg.model.mtp_num_layers
    )

    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 2
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)

    # Hash MoE layers must all live in the same pipeline stage as the embedding.
    # With a small num_layers / PP ratio the embedding stage may only have one
    # decoder layer, so cap moe_n_hash_layers accordingly.
    max_hash_layers = math.ceil(
        cfg.model.num_layers / cfg.model.pipeline_model_parallel_size
    )
    cfg.model.moe_n_hash_layers = min(
        getattr(cfg.model, "moe_n_hash_layers", max_hash_layers), max_hash_layers
    )

    # moe_layer_pattern is computed from moe_layer_freq. If moe_layer_freq is a
    # list, slice it to match the reduced num_layers.
    moe_layer_freq = getattr(cfg.model, "moe_layer_freq", None)
    if isinstance(moe_layer_freq, list):
        cfg.model.moe_layer_freq = moe_layer_freq[: cfg.model.num_layers]

    # linear_attention_pattern is computed from linear_attention_freq. Slice it
    # in the same way if it is a list.
    linear_attention_freq = getattr(cfg.model, "linear_attention_freq", None)
    if isinstance(linear_attention_freq, list):
        cfg.model.linear_attention_freq = linear_attention_freq[: cfg.model.num_layers]

    # Use the native cross entropy fusion implementation to avoid the TE fusion
    # stability warning rejected by Megatron-LM training args validation.
    #cfg.model.cross_entropy_fusion_impl = "native"

    cfg.dataset.seq_length = 256
    cfg.model.seq_length = cfg.dataset.seq_length

    cfg.model.hidden_size = 128
    cfg.model.num_attention_heads = 4

    cfg.model.num_moe_experts = 4
    cfg.model.moe_router_topk = 1

    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.train_iters = 20

    #AllReduce of CNCL has some problems
    cfg.ddp.average_in_collective = False

    # cfg.profiling = ProfilingConfig(
    #     use_nsys_profiler=True,
    #     profile_step_start=10,
    #     profile_step_end=15,
    #     profile_ranks=[0, 1],  # Profile first two ranks
    #     record_shapes=False,   # Optional: record tensor shapes
    # )
    cfg.profiling = ProfilingConfig(
        use_pytorch_profiler=True,
        profile_step_start=10,
        profile_step_end=15,
        profile_ranks=[0, 1],
        record_shapes=True,    # Record tensor shapes for detailed analysis
    )

    pretrain(cfg, forward_step)