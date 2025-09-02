

import torch
import torch.nn as nn
import torch.nn.functional as F




class DeepSeekMoEFFN(nn.Module):
    def __init__(self, d_model, hidden_size, num_experts=3, top_k=1):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_routing_experts = num_experts - 1  # 注意只 gating routing experts
        self.top_k = top_k

        # Router 只針對 routing experts（不包含 shared）
        self.gate = nn.Linear(d_model, self.num_routing_experts, bias=False)

        # Routing experts
        self.routing_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, d_model)
            )
            for _ in range(self.num_routing_experts)
        ])

        # Shared expert
        self.shared_expert = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, d_model)
        )

        self._balance_loss = 0.0  # routing balance loss

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        x_flat = x.view(-1, d_model)  # (batch * seq_len, d_model)

        ####紀錄oruting############
        self.last_input = x_flat

        # Gating for routing experts
        gate_scores = self.gate(x_flat)  # (batch*seq, num_routing_experts)
        gate_probs = F.softmax(gate_scores, dim=-1)

        # top-1 routing
        top1_idx = torch.argmax(gate_probs, dim=-1)  # (batch*seq,)
        one_hot = F.one_hot(top1_idx, num_classes=self.num_routing_experts).float()

        # balance loss
        mean_prob = gate_probs.mean(dim=0)
        self._balance_loss = -(mean_prob * torch.log(mean_prob + 1e-9)).sum()
        self._balance_loss = self._balance_loss / torch.log(torch.tensor(self.num_routing_experts, device=x.device))

        # dispatch to routing experts
        routing_outputs = []
        for i, expert in enumerate(self.routing_experts):
            mask = one_hot[:, i].unsqueeze(-1)
            expert_input = x_flat * mask
            expert_output = expert(expert_input)
            routing_outputs.append(expert_output)

        routing_outputs = torch.stack(routing_outputs, dim=0)  # (num_routing_experts, batch*seq, d_model)
        routing_output = routing_outputs.sum(dim=0)  # (batch*seq, d_model)

        # shared expert output
        shared_output = self.shared_expert(x_flat)

        # total output = routing + shared
        output = routing_output + shared_output

        return output.view(batch_size, seq_len, d_model)

    @property
    def balance_loss(self):
        return self._balance_loss







class MoETransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, activation="relu", batch_first=True, norm_first=True,num_experts=3, top_k=1):
        super().__init__()
        from torch.nn import TransformerDecoderLayer as TorchDecoderLayer
        # Attention 用官方原版
        self.self_attn = TorchDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first, norm_first).self_attn
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Feed Forward 改成 DeepSeekMoEFFN
        self.moe_ffn = DeepSeekMoEFFN(
            d_model=d_model,
            hidden_size=dim_feedforward,
            num_experts=num_experts,
            top_k=top_k
        )

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt
        x2 = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        x2 = self.moe_ffn(x)
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        return x, None

    @property
    def balance_loss(self):
        return self.moe_ffn.balance_loss
