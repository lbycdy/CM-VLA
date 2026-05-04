import math

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN


class MoEMLP(nn.Module): #共享专家内部
    def __init__(self, config, hidden_size = None, intermediate_size = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)) #标准 gated MLP(2个上投影+激活+逐元素乘法+下投影)

        return down_proj


class MoEGate_load_bal(nn.Module): #gate
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

        self.record_stats = False



        # 统计所有 suffix tokens，设成 0。
        self.stats_token_start = 0
        self.stats_token_end = None

        self.register_buffer(
            "expert_hard_count",
            torch.zeros(self.n_routed_experts, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "expert_weighted_count",
            torch.zeros(self.n_routed_experts, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "expert_soft_count",
            torch.zeros(self.n_routed_experts, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "expert_total_tokens",
            torch.zeros(1, dtype=torch.float64),
            persistent=False,
        )

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape        
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)  #对hidden_state做线性投影
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1) #再做softmax
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        self._update_expert_stats(topk_idx, topk_weight, scores, bsz, seq_len)

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0) #gate 给出的平均概率
                fi = ce * self.n_routed_experts #实际被选中的频率
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss

    @torch.no_grad()
    def reset_expert_stats(self):
        self.expert_hard_count.zero_()
        self.expert_weighted_count.zero_()
        self.expert_soft_count.zero_()
        self.expert_total_tokens.zero_()

    @torch.no_grad()
    def _update_expert_stats(self, topk_idx, topk_weight, scores, bsz, seq_len):
        """
        topk_idx:    [B * L, K]
        topk_weight: [B * L, K]
        scores:      [B * L, E]
        """
        if not self.record_stats:
            return

        num_experts = self.n_routed_experts
        top_k = self.top_k

        topk_idx = topk_idx.view(bsz, seq_len, top_k)
        topk_weight = topk_weight.view(bsz, seq_len, top_k).float()
        scores = scores.view(bsz, seq_len, num_experts).float()

        start = self.stats_token_start
        end = self.stats_token_end if self.stats_token_end is not None else seq_len

        topk_idx = topk_idx[:, start:end, :].reshape(-1)
        topk_weight = topk_weight[:, start:end, :].reshape(-1)
        scores = scores[:, start:end, :].reshape(-1, num_experts)

        if topk_idx.numel() == 0:
            return

        # 1) hard count：每个 token 的 top-k 专家都 +1
        hard = torch.bincount(
            topk_idx,
            minlength=num_experts,
        ).to(device=self.expert_hard_count.device, dtype=torch.float64)

        # 2) weighted count：每个 token 按 topk_weight 分配，总和约等于 token 数
        weighted = torch.bincount(
            topk_idx,
            weights=topk_weight.to(topk_idx.device),
            minlength=num_experts,
        ).to(device=self.expert_weighted_count.device, dtype=torch.float64)

        # 3) soft count：不只看 top-k，而是看 softmax 后所有 expert 的概率和
        soft = scores.sum(dim=0).to(
            device=self.expert_soft_count.device,
            dtype=torch.float64,
        )

        self.expert_hard_count += hard
        self.expert_weighted_count += weighted
        self.expert_soft_count += soft
        self.expert_total_tokens += scores.shape[0]


class MoEGate_mutual_info(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.condition_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        hidden_states = hidden_states.to(dtype=torch.bfloat16)
        bsz, seq_len, h = hidden_states.shape        
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            data_mask = hidden_states
            # Normalize datamask
            data_mask = F.normalize(data_mask, dim=-1)  # (N, d)

            # Normalize scores (treat as routing probability)
            routing_score = F.normalize(scores, dim=-1)  # (N, n_expert)

            # Compute similarity
            temperature = 0.1
            sim_matrix = routing_score @ routing_score.T  # (N, N)
            sim_matrix = sim_matrix / temperature

            # Compute pairwise datamask similarity (cosine)
            data_sim = data_mask @ data_mask.T           # (N, N)
            same_mask = data_sim > 0.99                  # threshold for identical datamask
            diag_mask = torch.eye(scores.shape[0], dtype=torch.bool, device=scores.device)
            same_mask = same_mask & (~diag_mask)

            # InfoNCE-style contrastive loss
            exp_sim = torch.exp(sim_matrix)
            numerator = (exp_sim * same_mask.float()).sum(dim=1)
            denominator = exp_sim.sum(dim=1)
            mi_loss = -torch.log(numerator / (denominator + 1e-8) + 1e-8)
            aux_loss = mi_loss.mean() * self.alpha
        else:
            aux_loss = None
        # print("MILoss:", aux_loss)
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss
    
    
class Cam2RobMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([MoEMLP(config, intermediate_size = config.moe_intermediate_size) for i in range(config.n_routed_experts)])
        self.gate = MoEGate_load_bal(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = MoEMLP(config=config, intermediate_size = intermediate_size)
    
    def forward(self, hidden_states, data_mask):
        identity = hidden_states  #保存残差信息和原始形状
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states) #gate 决定每个 token 去哪些 expert ，aux_loss 是 MoE 里的辅助损失
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1]) #把 token flatten，方便按 expert 分组处理
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0) #一个 token 要送到 4 个 expert
            y = torch.empty_like(hidden_states).float()
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i]).float() #按 expert 分桶
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1) #把每个 token 的 4 份 expert 输出按 topk_weight 加权求和。
            y =  y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss) #PyTorch 自定义自动求导函数 的调用方式，还是只拿到 y，但反向传播时 aux_loss 已经被算进去了。
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape) #把所有 token-expert 分配对按 id 排序，expert 输出乘上对应权重
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity) #加上 shared expert 输出
        return y
    
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x).float()
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).float()
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache


class ASMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([MoEMLP(config, intermediate_size = config.moe_intermediate_size) for i in range(config.n_routed_experts)])
        self.gate = MoEGate_mutual_info(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = MoEMLP(config=config, intermediate_size = intermediate_size)
    
    def forward(self, hidden_states, data_mask):
        identity = hidden_states
        orig_shape = hidden_states.shape
        data_mask = data_mask.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
        topk_idx, topk_weight, aux_loss = self.gate(data_mask)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i]).to(y.dtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y
    
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(dtype=expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache