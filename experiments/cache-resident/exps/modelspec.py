"""Qwen3.5 text-config facts and analytic byte counts (no torch needed).

Config values copied from the HF config.json of each repo (checked 2026-10-13).
"""
from __future__ import annotations

_COMMON = dict(attn_output_gate=True, full_attention_interval=4, head_dim=256, linear_conv_kernel_dim=4,
               linear_key_head_dim=128, linear_num_key_heads=16, linear_value_head_dim=128,
               num_attention_heads=8, num_key_value_heads=2, vocab_size=248320, tie_word_embeddings=True)

TEXT_CFG: dict[str, dict] = {
    "0.8B": dict(_COMMON, hidden_size=1024, intermediate_size=3584, num_hidden_layers=24, linear_num_value_heads=16),
    "2B": dict(_COMMON, hidden_size=2048, intermediate_size=6144, num_hidden_layers=24, linear_num_value_heads=16),
    "27B": dict(_COMMON, hidden_size=5120, intermediate_size=17408, num_hidden_layers=64, linear_num_value_heads=48,
                num_attention_heads=24, num_key_value_heads=4, tie_word_embeddings=False),
}


def _g(cfg, name, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def layer_types(cfg) -> list[str]:
    lt = _g(cfg, "layer_types")
    if lt:
        return list(lt)
    n, step = _g(cfg, "num_hidden_layers"), _g(cfg, "full_attention_interval", 4)
    return ["full_attention" if (i + 1) % step == 0 else "linear_attention" for i in range(n)]


def facts(cfg, dtype_bytes: int = 2) -> dict:
    """Analytic sizes for decode of one token. State is fp32, KV and weights use dtype_bytes."""
    lt = layer_types(cfg)
    n_gdn, n_full = lt.count("linear_attention"), lt.count("full_attention")
    hid, inter, vocab = _g(cfg, "hidden_size"), _g(cfg, "intermediate_size"), _g(cfg, "vocab_size")
    hk, hv = _g(cfg, "linear_num_key_heads"), _g(cfg, "linear_num_value_heads")
    dk, dv = _g(cfg, "linear_key_head_dim"), _g(cfg, "linear_value_head_dim")
    nh, nkv, hd = _g(cfg, "num_attention_heads"), _g(cfg, "num_key_value_heads"), _g(cfg, "head_dim")
    kern = _g(cfg, "linear_conv_kernel_dim", 4)
    gate = 2 if _g(cfg, "attn_output_gate", False) else 1
    key_dim, val_dim = hk * dk, hv * dv
    conv_dim = 2 * key_dim + val_dim
    gdn = conv_dim * hid + val_dim * hid + 2 * hv * hid + conv_dim * kern + 2 * hv + dv + hid * val_dim + 2 * hid
    attn = nh * hd * gate * hid + 2 * nkv * hd * hid + hid * nh * hd + 2 * hd + 2 * hid
    mlp = 3 * hid * inter
    body = n_gdn * (gdn + mlp) + n_full * (attn + mlp) + hid
    embed = vocab * hid
    return {
        "n_layers": len(lt), "n_gdn": n_gdn, "n_full": n_full, "hidden": hid, "vocab": vocab,
        "body_weight_bytes": body * dtype_bytes,
        "head_bytes": embed * dtype_bytes,
        "weight_bytes": (body + embed) * dtype_bytes,  # decode reads body + head once (embedding lookup is one row)
        "state_bytes": n_gdn * hv * dk * dv * 4,
        "conv_state_bytes": n_gdn * conv_dim * kern * dtype_bytes,
        "kv_bytes_per_token": n_full * 2 * nkv * hd * dtype_bytes,
    }
