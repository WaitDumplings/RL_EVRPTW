from dataclasses import dataclass

@dataclass
class Config:
    embed_dim = 128
    block_num = 8
    num_head = 8
    inpput_dim = 128
    attention_dim = 128
    decoder_head = 8
    tanh_clipping = 10
    temp = 1.0
    decode_type = "sampling"
    ffn_dim = 512