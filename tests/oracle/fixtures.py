"""Reference fixtures from the Python package (the oracle the Rust port is
checked against). Run with the needle repo's venv:

    ~/git/needle/.venv/bin/python tests/oracle/fixtures.py tokenizer|export|tiny|logits
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.expanduser("~/git/needle"))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(ROOT, "models")
OUT = os.path.join(ROOT, "tests", "oracle")


def tokenizer_cases():
    from needle.model.tokenizer import get_tokenizer
    tok = get_tokenizer()
    rng = random.Random(0)
    alphabet = list("abcdefghijklmnopqrstuvwxyz ABCXYZ0123456789.,:;!?'\"{}[]()<>/\\-_=+*&^%$#@~`|\n\t") + \
        ["é", "ü", "ñ", "日", "本", "語", "😀", "ß", "Ω", "▁", "​", "\x01"]
    markers = ["<|im_start|>", "<|im_end|>", "<tools>", "</tools>", "<tool_call>", "</tool_call>",
               "<think>", "</think>", "<image>", "<s>", "</s>", "<unk>", "<pad>", "<0x41>"]
    texts = ["", " ", "  ", "hello world", "hello  world", " leading", "trailing ",
             "set the living room lights to 30%", "email finance@cactus.dev, subject expenses",
             '<|im_start|>user\n<tools>[{"name":"set_lights","parameters":{"type":"object"}}]</tools>\ndim it<|im_end|>\n<|im_start|>assistant\n']
    for _ in range(3000):
        n = rng.randint(1, 60)
        parts = []
        for _ in range(n):
            r = rng.random()
            if r < 0.05:
                parts.append(rng.choice(markers))
            elif r < 0.35:
                parts.append(rng.choice(["the", " the", "lights", " room", "turn", "on", " off", "json", "name", "arguments"]))
            else:
                parts.append(rng.choice(alphabet))
        texts.append("".join(parts))
    cases = [{"text": t, "ids": tok.encode(t), "decoded": tok.decode(tok.encode(t))} for t in texts]
    with open(os.path.join(OUT, "tokenizer_cases.json"), "w") as f:
        json.dump(cases, f, ensure_ascii=False)
    print(len(cases), "tokenizer cases")


def export_full():
    from needle.model.run import load_checkpoint
    from needle.model.export import write_export
    from needle.model.architecture import effective_kv_window
    from needle.model.tokenizer import get_tokenizer
    params, config = load_checkpoint(os.path.join(MODELS, "needle3.safetensors"))
    out = os.path.join(MODELS, "needle3_py4.cact")
    info = write_export(params, config, out, bits=4, tokenizer=get_tokenizer(config.vocab_size),
                        kv_window=effective_kv_window(config))
    print(info)



QUERY = "dim the living room lights to 30 and tell me the weather in Lagos"
TOOLS = [
    {"name": "set_lights", "description": "Turn a room's lights on/off and set brightness",
     "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "on": {"type": "boolean"},
                                                      "brightness": {"type": "integer", "minimum": 0, "maximum": 100}},
                    "required": ["room", "on"]}},
    {"name": "get_weather", "description": "Get the current weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
]


def _save(name, arr):
    import numpy as np
    arr = np.ascontiguousarray(np.asarray(arr, np.float32))
    arr.tofile(os.path.join(MODELS, name + ".f32"))
    return list(arr.shape)


def logits():
    """Full-model reference logits in float32, plain and under the finetune
    numerics (CQ W4 STE weights + A8 activations + int8 KV)."""
    import time
    import numpy as np
    import jax
    import jax.numpy as jnp
    from needle.model.run import load_checkpoint, build_prompt
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.tokenizer import get_tokenizer, BOS_ID
    from needle.model.quantize import configure_deploy, cq_ste_params, WEIGHT_BITS

    params, config = load_checkpoint(os.path.join(MODELS, "needle3.safetensors"))
    config.dtype = "float32"
    params = jax.tree.map(lambda a: jnp.asarray(np.asarray(a, np.float32)), params)
    tok = get_tokenizer()
    prompt = build_prompt(QUERY, TOOLS)
    ids = [BOS_ID] + tok.encode(prompt)
    model = SimpleAttentionNetwork(config)
    x = jnp.asarray([ids], jnp.int32)
    fwd = jax.jit(lambda p, t: model.apply({"params": p}, t))
    out = fwd(params, x); out.block_until_ready()
    t0 = time.time(); out = fwd(params, x); out.block_until_ready(); dt = time.time() - t0
    meta = {"ids": ids, "prompt": prompt, "logits": _save("oracle_logits", out[0]), "forward_s": dt}

    configure_deploy(act_bits=8, kv_bits=8)
    qfwd = jax.jit(lambda p, t: model.apply({"params": cq_ste_params(p, WEIGHT_BITS)}, t, quant=True))
    qout = qfwd(params, x); qout.block_until_ready()
    meta["qlogits"] = _save("oracle_qlogits", qout[0])
    conf = jax.jit(lambda p, t: model.apply({"params": p}, t, method=model.forward_confidence))(params, x)
    meta["confidence_logit"] = float(conf[0])
    with open(os.path.join(OUT, "logits.json"), "w") as f:
        json.dump(meta, f)
    print("seq", len(ids), "forward", dt, "conf", float(conf[0]))


FIXTURES = {"tokenizer": tokenizer_cases, "export": export_full, "logits": logits}



def split_quant():
    """Separate the finetune numerics: CQ weights alone, A8/KV8 alone."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from needle.model.run import load_checkpoint
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.quantize import configure_deploy, cq_ste_params, WEIGHT_BITS
    meta = json.load(open(os.path.join(OUT, "logits.json")))
    params, config = load_checkpoint(os.path.join(MODELS, "needle3.safetensors"))
    config.dtype = "float32"
    params = jax.tree.map(lambda a: jnp.asarray(np.asarray(a, np.float32)), params)
    model = SimpleAttentionNetwork(config)
    x = jnp.asarray([meta["ids"]], jnp.int32)
    configure_deploy(act_bits=8, kv_bits=8)
    w = jax.jit(lambda p, t: model.apply({"params": cq_ste_params(p, WEIGHT_BITS)}, t))(params, x)
    a = jax.jit(lambda p, t: model.apply({"params": p}, t, quant=True))(params, x)
    meta["wq_logits"] = _save("oracle_wq_logits", w[0])
    meta["aq_logits"] = _save("oracle_aq_logits", a[0])
    json.dump(meta, open(os.path.join(OUT, "logits.json"), "w"))


FIXTURES["split_quant"] = split_quant



def cells():
    """Per-layer lane-mean hidden cells, float and quant."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from needle.model.run import load_checkpoint
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.quantize import configure_deploy
    meta = json.load(open(os.path.join(OUT, "logits.json")))
    params, config = load_checkpoint(os.path.join(MODELS, "needle3.safetensors"))
    config.dtype = "float32"
    params = jax.tree.map(lambda a: jnp.asarray(np.asarray(a, np.float32)), params)
    model = SimpleAttentionNetwork(config)
    x = jnp.asarray([meta["ids"]], jnp.int32)
    configure_deploy(act_bits=8, kv_bits=8)
    f = jax.jit(lambda p, t: model.apply({"params": p}, t, method=model.hidden_cells))(params, x)
    q = jax.jit(lambda p, t: model.apply({"params": p}, t, quant=True, method=model.hidden_cells))(params, x)
    meta["cells"] = _save("oracle_cells", f[0])
    meta["qcells"] = _save("oracle_qcells", q[0])
    json.dump(meta, open(os.path.join(OUT, "logits.json"), "w"))


FIXTURES["cells"] = cells



def grads():
    """Loss and LoRA gradients from jax.value_and_grad on a real batch, at a
    nonzero LoRA state, plus the reference LoRA init for seed 0."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    import optax
    from needle.model.run import load_checkpoint
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.tokenizer import get_tokenizer
    from needle.model.quantize import configure_deploy, cq_ste_params, WEIGHT_BITS
    from needle.model.finetune import (lora_target_paths, init_lora, merge_lora, load_jsonl)
    from needle.model.checkpoints import write_adapter
    params, config = load_checkpoint(os.path.join(MODELS, "needle3.safetensors"))
    config.dtype = "float32"
    params = jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params)
    model = SimpleAttentionNetwork(config)
    configure_deploy(act_bits=8, kv_bits=8)
    tok = get_tokenizer()
    paths = lora_target_paths(params)
    print("target order", ["/".join(p) for p in paths])
    lora0 = init_lora(params, paths, 16, jax.random.PRNGKey(0))
    write_adapter(os.path.join(MODELS, "oracle_lora_init.safetensors"), {
        "lora": {"/".join(p): {"A": np.asarray(v["A"]), "B": np.asarray(v["B"])} for p, v in lora0.items()},
        "scale": 2.0, "base": "x", "rank": 16, "seed": 0})
    rng = np.random.default_rng(123)
    lora = {p: {"A": v["A"], "B": jnp.asarray(rng.normal(size=v["B"].shape).astype(np.float32) * 0.01)}
            for p, v in lora0.items()}
    data = os.path.join(OUT, "grad_batch.jsonl")
    seqs, masks = load_jsonl(data, tok, 1024)
    print("batch", seqs.shape)

    def loss_fn(lora, ids, mask):
        merged = cq_ste_params(merge_lora(params, lora, 2.0), WEIGHT_BITS)
        logits = model.apply({"params": merged}, ids, quant=True)
        logits, targets, mask = logits[:, :-1], ids[:, 1:], mask[:, 1:]
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return (ce * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    loss, g = jax.jit(jax.value_and_grad(loss_fn))(lora, jnp.asarray(seqs), jnp.asarray(masks))
    loss0 = jax.jit(loss_fn)(lora0, jnp.asarray(seqs), jnp.asarray(masks))
    write_adapter(os.path.join(MODELS, "oracle_lora_state.safetensors"), {
        "lora": {"/".join(p): {"A": np.asarray(v["A"]), "B": np.asarray(v["B"])} for p, v in lora.items()},
        "scale": 2.0, "base": "x", "rank": 16, "seed": 0})
    write_adapter(os.path.join(MODELS, "oracle_lora_grads.safetensors"), {
        "lora": {"/".join(p): {"A": np.asarray(v["A"]), "B": np.asarray(v["B"])} for p, v in g.items()},
        "scale": 2.0, "base": "x", "rank": 16, "seed": 0})
    json.dump({"loss": float(loss), "loss_init": float(loss0)}, open(os.path.join(OUT, "grads.json"), "w"))
    print("loss", float(loss), "loss at init", float(loss0))


FIXTURES["grads"] = grads



def float_grads():
    """As `grads`, float numerics (quant off): the backward pass itself."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    import optax
    from needle.model.run import load_checkpoint
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.tokenizer import get_tokenizer
    from needle.model.finetune import merge_lora, load_jsonl
    from needle.model.checkpoints import write_adapter, read_adapter
    params, config = load_checkpoint(os.path.join(MODELS, "needle3.safetensors"))
    config.dtype = "float32"
    params = jax.tree.map(lambda a: jnp.asarray(np.asarray(a).astype(np.float32)), params)
    model = SimpleAttentionNetwork(config)
    tok = get_tokenizer()
    ad = read_adapter(os.path.join(MODELS, "oracle_lora_state.safetensors"))
    lora = {tuple(k.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])} for k, v in ad["lora"].items()}
    seqs, masks = load_jsonl(os.path.join(OUT, "grad_batch.jsonl"), tok, 1024)

    def loss_fn(lora, ids, mask):
        merged = merge_lora(params, lora, 2.0)
        logits = model.apply({"params": merged}, ids)
        logits, targets, mask = logits[:, :-1], ids[:, 1:], mask[:, 1:]
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return (ce * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    loss, g = jax.jit(jax.value_and_grad(loss_fn))(lora, jnp.asarray(seqs), jnp.asarray(masks))
    write_adapter(os.path.join(MODELS, "oracle_lora_fgrads.safetensors"), {
        "lora": {"/".join(p): {"A": np.asarray(v["A"]), "B": np.asarray(v["B"])} for p, v in g.items()},
        "scale": 2.0, "base": "x", "rank": 16, "seed": 0})
    meta = json.load(open(os.path.join(OUT, "grads.json")))
    meta["float_loss"] = float(loss)
    json.dump(meta, open(os.path.join(OUT, "grads.json"), "w"))
    print("float loss", float(loss))


FIXTURES["float_grads"] = float_grads


if __name__ == "__main__":
    FIXTURES[sys.argv[1]]()
