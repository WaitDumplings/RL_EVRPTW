import torch
import torch.nn as nn
import math
from configuration import Config

# Skip Connection Module
class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)

# Normalization Module
class Normalization(nn.Module):
    """
    Normalization module to apply LayerNorm, BatchNorm1d, or InstanceNorm1d.

    Args:
        embed_dim (int): Embedding dimension for normalization.
        normalization (str): Type of normalization ('layer', 'batch', 'instance').

    Returns:
        Normalized tensor of the same shape as input.
    """

    def __init__(self, embed_dim, normalization='batch'):
        super().__init__()
        if normalization == 'layer':
            self.normalizer = nn.LayerNorm(embed_dim)
        elif normalization == 'batch':
            self.normalizer = nn.BatchNorm1d(embed_dim)
        elif normalization == "instance":
            self.normalizer = nn.InstanceNorm1d(embed_dim)
        else:
            raise ValueError(f"Unsupported normalization type: {normalization}")

    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        return self.normalizer(x)

# Feed-Forward Network with Residuals
class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=nn.ReLU, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.residual = SkipConnection(self.ffn)

    def forward(self, x):
        return self.residual(x)

# Multi-Head Attention Module
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim):
        super().__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.proj_qkv = nn.Linear(input_dim, embed_dim * 3)
        self.proj_out = nn.Linear(embed_dim, input_dim)
        self.norm_factor = 1 / math.sqrt(embed_dim // n_heads)

    def forward(self, x, mask=None):
        assert len(x.size()) == 3, "The input shape should follow the rule: Batch Size, Seq_len, Hidden dim"
        bs, seq_len, _ = x.size()
        qkv = self.proj_qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(bs, seq_len, self.n_heads, -1).transpose(1, 2) for t in qkv]

        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.norm_factor
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bs, seq_len, -1)
        return self.proj_out(out)

# Multi-Head Attention Block
class MHA_Block(nn.Module):
    def __init__(self, num_head, input_dim, attention_dim, ffn_dim, normalization='batch', dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(num_head, input_dim, attention_dim)
        self.norm1 = Normalization(input_dim, normalization)
        self.ffn = FFN(input_dim, ffn_dim, input_dim, dropout=dropout)
        self.norm2 = Normalization(input_dim, normalization)

    def forward(self, x, mask=None):
        # Multi-Head Attention with Residual Connection and Normalization
        x = self.norm1(x + self.mha(x, mask=mask))
        # Feed-Forward Network with Residual Connection and Normalization
        x = self.norm2(x + self.ffn(x))
        return x

# Embedding Layer
class Embedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.deport_embedding = nn.Linear(2, embed_dim)
        self.nodes_embedding = nn.Linear(3, embed_dim)

    def forward(self, x):
        deport = self.deport_embedding(x['deport'])
        if len(x['demand'].shape) != len(x['nodes'].shape):
            x['demand'] = x['demand'].unsqueeze(-1)
        nodes = self.nodes_embedding(torch.cat((x['nodes'], x['demand']), -1))

        return torch.cat((deport, nodes), dim=-2)

# Attention Model with Configurable Layers
class Encoder(nn.Module):
    """
    Attention-based model with multiple MHA blocks.

    Args:
        block_num (int): Number of multi-head attention blocks.
        num_head (int): Number of attention heads.
        input_dim (int): Input feature dimension.
        attention_dim (int): Total attention dimension.
        ffn_dim (int): Hidden dimension in feed-forward network.
        normalization (str): Type of normalization ('layer' or 'batch').
        dropout (float): Dropout probability.
    """
    def __init__(self, block_num, num_head, input_dim, attention_dim, ffn_dim, normalization='layer', dropout=0.1):
        super().__init__()
        self.attention_layers = nn.ModuleList([
            MHA_Block(num_head, input_dim, attention_dim, ffn_dim, normalization, dropout)
            for _ in range(block_num)
        ])

    def forward(self, x, mask=None):
        for layer in self.attention_layers:
            x = layer(x, mask=mask)

        return x
    
class Decoder(nn.Module):
    def __init__(self, input_dim, attention_dim, decoder_head, tanh_clipping, decode_type, env, mask_logits = True, temp = 1.0, normalization='batch', dropout=0.1):
        super().__init__()
        self.decoder_head = decoder_head
        self.attention_dim = attention_dim
        self.norm_factor = 1 / math.sqrt(attention_dim)
        self.tanh_clipping = tanh_clipping
        self.env = env
        self.mask_logits = mask_logits
        self.temp = temp
        self.decode_type = decode_type

        self.time_embedding = nn.Linear(1, attention_dim // decoder_head, bias=False)
        self.context_project = nn.Linear(input_dim, attention_dim, bias=False)
        self.kvlogit_project = nn.Linear(input_dim, attention_dim * 3, bias=False)
        self.project_step_context = nn.Linear(input_dim + 1, attention_dim, bias=False)
        self.project_out = nn.Linear(attention_dim, attention_dim, bias=False)

    def _select_node(self, probs, mask):
        breakpoint()
        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected


    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        batch_size, _, embed_dim = query.size()
        key_size = val_size = embed_dim // self.decoder_head

        # Get Glimpse Q
        glimpse_Q = query.view(batch_size, self.decoder_head, 1, key_size)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if mask:
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, embedding_dim)
        glimpse = self.project_out(
            heads.transpose(1, 2).contiguous().view(-1, 1, self.decoder_head * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)

        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits and mask is not None:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_log_p(self, embeddings, context_node, glimpse_K, glimpse_V, logit_K, state, normalize=True):
        batch_size = glimpse_K.shape[0]
        current_node = state.get_current_node() if state else torch.zeros(batch_size, 1, dtype=torch.int64)
        next_nodes = torch.cat((torch.gather(embeddings, 1, current_node.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, embeddings.size(-1))).view(batch_size, 1, embeddings.size(-1)), self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None]),-1)
        # next_nodes = torch.cat((torch.gather(embeddings, 1, current_node.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, embeddings.size(-1))).view(batch_size, 1, embeddings.size(-1)), torch.ones_like(current_node).unsqueeze(-1)),-1)
        
        # Compute query = context node embedding
        query = context_node + \
                self.project_step_context(next_nodes)

        # Compute the mask
        mask = state.get_mask() if state else None

        # time embedding (cur_time: bs, 1, 16)
        current_time = state.get_current_time() if state is not None else torch.zeros(batch_size, 1).unsqueeze(1)
        time_embedding = self.time_embedding(current_time).unsqueeze(1)
        glimpse_K += time_embedding
        glimpse_V += time_embedding

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def forward(self, env, embedding, mask=None):
        outputs = []
        sequences = []

        # avg hidden states to get graph embedding
        # BR1
        graph_embedding = embedding.mean(1)
        fixed_context = self.context_project(graph_embedding).unsqueeze(1)

        # Get glimps KV, logits_k
        # BR2
        glimps_k, glimps_v, logits_k = self.kvlogit_project(embedding).chunk(3, dim=-1)
        batch_size, seq_len, hidden_dim = glimps_k.size()

        assert (hidden_dim%self.decoder_head)==0, "The dim of decoder cannot match its decoder head!"
        glimps_k = glimps_k.view(batch_size, seq_len, self.decoder_head, hidden_dim//self.decoder_head).contiguous().transpose(1, 2)
        glimps_v = glimps_v.view(batch_size, seq_len, self.decoder_head, hidden_dim//self.decoder_head).contiguous().transpose(1, 2)
        logits_k = logits_k.contiguous()

        # Reset env
        state = env.reset() if env is not None else None

        # Perform decoding steps
        i = 0
        # while not state.all_finished():
        while i < 2:
            log_p, mask = self._get_log_p(embedding, fixed_context, glimps_k, glimps_v, logits_k, state)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            mask = torch.zeros_like(log_p.exp())
            selected = self._select_node(log_p.exp(), mask)  # Squeeze out steps dimension

            state = state.update(selected)

            # Now make log_p, selected desired output size by 'unshrinking'
            if state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

class EVRPTW_Model(nn.Module):
    def __init__(self, config, env):
        super().__init__()
        self.embedding = Embedding(config.embed_dim)
        self.encoder = Encoder(block_num = config.block_num,
                               num_head = config.num_head,
                               input_dim = config.embed_dim,
                               attention_dim = config.attention_dim,
                               ffn_dim = config.ffn_dim,
                               normalization = "batch")

        self.decoder = Decoder(input_dim = config.embed_dim,
                               attention_dim = config.attention_dim,
                               decoder_head = config.decoder_head,
                               tanh_clipping = config.tanh_clipping,
                               temp = config.temp,
                               decode_type = config.decode_type,
                               env = env)
        self.env = env
    
    def forward(self, dict_x):
        x = self.embedding(dict_x)
        x = self.encoder(x)
        breakpoint()
        x = self.decoder(env = self.env,
                         embedding = x)
        return x

# Weight Initialization Function
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

# Example Usage
if __name__ == "__main__":
    config = Config()
    env = None
    model = EVRPTW_Model(config, env)
    model.apply(init_weights)  # Apply weight initialization

    dict_x = {"deport": torch.rand(10, 1, 2), "nodes": torch.rand(10, 100, 2), "demand": torch.rand(10, 100)}
    model(dict_x)
    # input_data = torch.rand(10, 15, 128)  # (batch_size, seq_len, input_dim)
    # output = model(input_data)
    # print(output.shape)  # Should be (10, 15, 128)
