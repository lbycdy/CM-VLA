import torch
import torch.distributed as dist

from src.openpi.models.modeling_moe import MoEGate_load_bal


def enable_moe_stats(model, enabled=True, action_tokens_only=False):
    """
    action_tokens_only=True:  跳过前 2 个 token，只统计 action tokens
    """
    for name, module in model.named_modules():
        if isinstance(module, MoEGate_load_bal):
            module.record_stats = enabled
            module.stats_token_start = 2 if action_tokens_only else 0
            module.stats_token_end = None


def reset_moe_stats(model):
    for module in model.modules():
        if isinstance(module, MoEGate_load_bal):
            module.reset_expert_stats()


@torch.no_grad()
def collect_moe_stats(model, reduce_distributed=True):
    """
    返回每一层 gate 的统计结果。
    """
    results = {}

    for name, module in model.named_modules():
        if not isinstance(module, MoEGate_load_bal):
            continue

        hard = module.expert_hard_count.detach().clone()
        weighted = module.expert_weighted_count.detach().clone()
        soft = module.expert_soft_count.detach().clone()
        total_tokens = module.expert_total_tokens.detach().clone()

        if (
            reduce_distributed
            and dist.is_available()
            and dist.is_initialized()
        ):
            dist.all_reduce(hard, op=dist.ReduceOp.SUM)
            dist.all_reduce(weighted, op=dist.ReduceOp.SUM)
            dist.all_reduce(soft, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

        results[name] = {
            "hard_count": hard.cpu(),
            "weighted_count": weighted.cpu(),
            "soft_count": soft.cpu(),
            "total_tokens": total_tokens.cpu(),
        }

    return results


def aggregate_expert_stats(stats, mode="weighted_count"):
    """
    把所有 MoE 层加起来，得到最终一行 expert activation。
    mode:
      - hard_count: top-k 命中次数，行总和 = token数 * top_k
      - weighted_count: 按 topk_weight 加权，行总和 ≈ token数，最推荐画热力图
      - soft_count: softmax 概率总和，不限 top-k
    """
    rows = []

    for layer_name, item in stats.items():
        rows.append(item[mode].float())

    if len(rows) == 0:
        raise RuntimeError("No MoEGate_load_bal modules were found.")

    total = torch.stack(rows, dim=0).sum(dim=0)
    ratio = total / (total.sum() + 1e-12)

    return {
        "count": total,
        "ratio": ratio,
    }