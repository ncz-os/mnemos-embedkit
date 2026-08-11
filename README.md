> # 📍 Moved to GitLab
> **The canonical, authoritative home of this project is GitLab — always:**
> ## 👉 https://gitlab.com/ncz-os/mnemos-embedkit
>
> This GitHub repository is a **frozen, read-only mirror**. All development, issues, and releases happen on GitLab. Please open issues and merge requests there. The full history of this stub is preserved on GitLab.

---

# mnemos-embedkit

> Open embedding devkit. Same API across supported runtimes.

`embedkit` lets you embed text once and run it through one of the adapters implemented today: Cix Sky1 NPU, NVIDIA CUDA through ONNX Runtime, or llama.cpp on CPU. The kit detects installed runtimes and picks the highest-tier adapter that can load the model you provide. **No vendor preference.**

## Quick start

```python
import embedkit

eng = embedkit.Engine.auto(model="bge-small-zh-v1.5")
vec = eng.embed("Hello world")            # -> List[float]
vecs = eng.embed_batch(["a", "b", "c"])   # -> List[List[float]]

eng.info()
# {"adapter": "cix-npu", "model": "bge-small-zh-v1.5_256.cix",
#  "embed_dim": 512, "max_tokens": 256, "throughput_baseline": 55.0}
```

Explicit adapter pick:

```python
eng = embedkit.Engine(adapter="cix-npu", model="/models/bge-small-zh-v1.5/model.cix")
eng = embedkit.Engine(adapter="nvidia-cuda", model="/models/bge-small-zh-v1.5/model.onnx")
eng = embedkit.Engine(adapter="cpu-llamacpp", model="/models/bge-small-zh-v1.5/model.gguf")
```

## What the kit is

A pure-Python adapter layer over vendor-specific embedding runtimes, plus a uniform `Engine.embed*` API and a canonical bench harness. **The kit does not bundle drivers or kernel modules.** It detects what the host OS already provides and binds to it:

| Host has | Kit picks via |
|---|---|
| `cix-noe-umd 2.0.2` + `libnoe` (NCZ Magnetar / cixtech apt) | `npu-cix` adapter |
| `onnxruntime-gpu` (CUDA driver from Linux distro) | `nvidia-cuda` adapter |
| `llama-cpp-python` | `cpu-llamacpp` adapter |

Models are not bundled or downloaded. Pass a path to a model provisioned for the selected adapter. Short model names resolve beneath `EMBEDKIT_MODELS_DIR` (default `/opt/ncz/models`) and must already exist.

## Install

```bash
# Pick the form-factor bundle that matches your host:
pip install "mnemos-embedkit[all-cpu] @ git+https://gitlab.com/ncz-os/mnemos-embedkit.git"
pip install "mnemos-embedkit[all-x86-cuda] @ git+https://gitlab.com/ncz-os/mnemos-embedkit.git"
pip install "mnemos-embedkit[all-arm-cix] @ git+https://gitlab.com/ncz-os/mnemos-embedkit.git"
pip install "mnemos-embedkit[all] @ git+https://gitlab.com/ncz-os/mnemos-embedkit.git"
```

The kit pulls vendor *python bindings* from PyPI. Vendor *drivers* are managed by your OS package manager (for example, cix-noe-umd via apt or an NVIDIA driver supplied by the Linux distribution).

## Reference bench

The included smoke corpus is in `benches/`. From a source checkout, run:

```bash
embedkit-bench --corpus benches/corpora/smoke.json --engines auto --model bge-small-zh-v1.5
```

See `benches/results.md` for the cross-platform numbers we have today (Cix Sky1 NPU, Apple Silicon Metal, NVIDIA CUDA, x86 + ARM CPU, Pi 5, Pi 4).

## Reference implementation consumer

MNEMOS does not currently import this package. Integration with `ncz-os/mnemos` remains planned work; installing Embedkit alone does not change MNEMOS ingest routing.

## License

Apache-2.0.

## Status

**Pre-alpha.** CPU llama.cpp, NVIDIA CUDA, and Cix NPU adapters are implemented. Other backends described in the design documents remain roadmap items and are not installable extras.

See `docs/DESIGN.md` for the full architecture.


## Build infrastructure & partners

Continuous integration and package distribution for this project are generously
supported by our open-source infrastructure partners:

- **[GitLab](https://gitlab.com/)** — canonical source hosting and CI pipelines
  (format / lint / test gates), via the
  [GitLab for Open Source](https://about.gitlab.com/solutions/open-source/) program.
- **[Buildkite](https://buildkite.com/)** — CI/CD orchestration with hosted macOS
  and Linux agents, and our APT package registry host
  (`packages.buildkite.com/ncz-os/ncz`), via the
  [Buildkite Open Source](https://buildkite.com/pricing) program.

Thank you to both for backing open-source software.
