# moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class Top2Gate(nn.Module):
    def __init__(self, input_dim, num_routed_experts=3, top_k=1):
        super().__init__()
        self.num_routed_experts = num_routed_experts
        self.total_experts = num_routed_experts + 1  # +1 for shared expert
        self.top_k = top_k
        self.w_gating = nn.Linear(input_dim, num_routed_experts, bias=False)  # Only route to routed experts

    def forward(self, x):
        # Get routing scores only for routed experts
        logits = self.w_gating(x)  # [B, T, num_routed_experts]

        ## Gumbel noise
        noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
        logits = logits + gumbel_noise       

        scores = F.softmax(logits, dim=-1)  # [B, T, num_routed_experts]

        ###觀察gating結果
        # entropy = -(scores * scores.clamp(min=1e-9).log()).sum(dim=-1)  # [B, T]
        # avg_entropy = entropy.mean()
        


        # Select top-k from routed experts
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)  # [B, T, k]

        # Create dispatch mask for all experts (including shared)
        dispatch_mask = torch.zeros(x.shape[0], x.shape[1], self.total_experts, device=x.device)  # [B, T, total_experts]
        
        # Set mask for routed experts (indices need to be offset by 1 to account for shared expert)
        for k in range(self.top_k):
            # Add 1 to indices to skip shared expert (index 0)
            dispatch_mask.scatter_(-1, topk_indices[..., k:k+1] + 1, topk_scores[..., k:k+1])
        
        # Set mask for shared expert (always 1.0)
        dispatch_mask[..., 0] = 1.0  # Shared expert always gets full weight

        # Load balancing loss (only over routed experts)
        me = scores.mean(dim=[0, 1])  # [num_routed_experts]
        ce = (scores ** 2).mean(dim=[0, 1])  # [num_routed_experts]
        load_balance_loss = (me * ce).sum() * (self.num_routed_experts ** 2)


        #  Debug: 印出第一個 token 的 routing 分數與 load balance 狀況
        # if self.training:
        #     with torch.no_grad():
        #         print("Top2Gate DEBUG INFO:")
        #         print("scores[0, 0] =", scores[0, 0].detach().cpu().numpy())
        #         print("mean usage per expert =", me.detach().cpu().numpy())
        #         print("load_balance_loss =", load_balance_loss.item())


        return dispatch_mask, load_balance_loss



class MoELayer(nn.Module):####num_routed_experts=4 , top_k=2,
    def __init__(self, input_size, hidden_size, num_routed_experts=3, top_k=1, routed_expert_scale=0.5):
        """
        Args:
            input_size: Input dimension
            hidden_size: Hidden size for shared expert (will be used as base size)
            num_routed_experts: Number of routed experts
            top_k: Number of experts to route to
            routed_expert_scale: Scale factor for routed experts' hidden size (relative to shared expert)
        """
        super().__init__()
        self.total_experts = num_routed_experts + 1  # 1 shared + num_routed_experts routed
        self.gate = Top2Gate(input_size, num_routed_experts, top_k)
        
        # Calculate routed expert hidden size
        routed_hidden_size = int(hidden_size * routed_expert_scale)
        
        # Expert networks
        self.experts = nn.ModuleList([
            # Shared expert (index 0) - full size
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_size),
            )
        ] + [
            # Routed experts - smaller size
            nn.Sequential(
                nn.Linear(input_size, routed_hidden_size),
                nn.ReLU(),
                nn.Linear(routed_hidden_size, input_size),
            ) for _ in range(num_routed_experts)
        ])

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        dispatch_mask, load_balance_loss = self.gate(x)  # [B, T, total_experts]

        expert_outputs = torch.zeros_like(x)  # Initialize directly to save memory

        for i, expert in enumerate(self.experts):
            # Get the mask for current expert
            expert_mask = dispatch_mask[..., i]  # [B, T]

            # Find the positions where the mask is non-zero
            tokens_to_compute = expert_mask.nonzero(as_tuple=True)  # tuple of indices

            if tokens_to_compute[0].numel() > 0:
                # Extract the corresponding inputs
                input_to_expert = x[tokens_to_compute]  # [num_tokens, D]

                # Compute expert output
                computed_output = expert(input_to_expert)  # [num_tokens, D]

                # Scale by the corresponding mask values (weights)
                weights = expert_mask[tokens_to_compute].unsqueeze(-1)  # [num_tokens, 1]
                scaled_output = computed_output * weights  # [num_tokens, D]

                # Assign the computed values back to the expert_outputs tensor
                expert_outputs[tokens_to_compute] += scaled_output

        return expert_outputs, load_balance_loss

