import pytest

from jarvis_rag.check_env import cosine, estimate_vram_mib, model_arch, parse_nvidia_smi

# Trimmed /api/show response for qwen3:4b-instruct, as returned by Ollama 0.40.
QWEN3_4B_SHOW = {
    "details": {"quantization_level": "Q4_K_M"},
    "model_info": {
        "general.architecture": "qwen3",
        "general.parameter_count": 4022468096,
        "qwen3.block_count": 36,
        "qwen3.attention.head_count": 32,
        "qwen3.attention.head_count_kv": 8,
        "qwen3.attention.key_length": 128,
        "qwen3.embedding_length": 2560,
    },
}


def test_parse_nvidia_smi():
    gpus = parse_nvidia_smi("NVIDIA GeForce GTX 1650, 4096, 3, 595.91.07\n")
    assert gpus == [{"name": "NVIDIA GeForce GTX 1650", "total_mib": 4096, "used_mib": 3, "driver": "595.91.07"}]


def test_model_arch_from_ollama_show():
    arch = model_arch(QWEN3_4B_SHOW)
    assert arch == {"n_params": 4022468096, "quant": "Q4_K_M", "n_layers": 36, "n_kv_heads": 8, "head_dim": 128}


def test_model_arch_derives_head_dim_when_key_length_missing():
    show = {**QWEN3_4B_SHOW, "model_info": dict(QWEN3_4B_SHOW["model_info"])}
    del show["model_info"]["qwen3.attention.key_length"]
    assert model_arch(show)["head_dim"] == 2560 // 32


def test_kv_cache_matches_llama_cpp_log():
    # llama.cpp logged "llama_kv_cache: size = 576.00 MiB (4096 cells, 36 layers)"
    # for qwen3:4b with an f16 cache. The formula should reproduce it exactly.
    arch = model_arch(QWEN3_4B_SHOW)
    est = estimate_vram_mib(arch["n_params"], 4.85, arch["n_layers"], arch["n_kv_heads"], arch["head_dim"], 4096)
    assert est["kv_cache"] == pytest.approx(576.0)
    assert est["weights"] == pytest.approx(2326, abs=1)
    assert est["total"] == pytest.approx(est["weights"] + est["kv_cache"])


def test_kv_cache_scales_linearly_with_context_and_precision():
    base = estimate_vram_mib(4_000_000_000, 4.85, 36, 8, 128, 4096)["kv_cache"]
    assert estimate_vram_mib(4_000_000_000, 4.85, 36, 8, 128, 8192)["kv_cache"] == pytest.approx(2 * base)
    assert estimate_vram_mib(4_000_000_000, 4.85, 36, 8, 128, 4096, kv_bytes=1)["kv_cache"] == pytest.approx(base / 2)


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([1, 1], [-1, -1]) == pytest.approx(-1.0)
