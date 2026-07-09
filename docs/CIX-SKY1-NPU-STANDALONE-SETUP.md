# cix-npu adapter — standalone setup (non-NCZ-OS)

This guide covers running the `cix-npu` adapter on a CIX Sky1 board (Radxa
Orion O6, MS-R1, or similar) **without** NCZ-OS. On NCZ-OS this whole setup
is baked into the ISO by `cix-installer/post-install/47-embedkit.sh` and
`25-cix-proprietary.sh` — this doc reproduces that path by hand for any
other distro on the same SoC.

`embedkit` itself is not Cix-specific (see the main README for the
NVIDIA/AMD/Apple/Intel/Rockchip/MediaTek adapters) — this page is scoped to
just the Cix Sky1 NPU (Zhouyi) path.

## Prerequisites

- CIX Sky1 board with the NPU kernel driver already loaded (`/dev/aipu`
  present). This is a kernel-level concern, independent of embedkit — see
  your distro's NPU driver docs.
- Docker (if running MNEMOS via container) or any host to run
  `mnemos-embedkit` against.
- **Python >= 3.11.** `libnoe`/`NOE_Engine` as shipped in UMD 2.0.x are
  `py3-none` (pure-Python-tagged) wheels, not interpreter-pinned — check
  `ls /usr/share/cix/lib` after step 2 (below) for the actual filenames on
  your box rather than assuming a cp3xx tag.

## ⚠ Not on PyPI yet

`mnemos-embedkit` is Pre-Alpha and **not published to PyPI**.
`pip install mnemos-embedkit` will 404. Worse: plain `pip install embedkit`
silently resolves to a **different, unrelated package** someone else owns
on PyPI — it is not this project and has no `Engine`/`cix-npu` adapter.
Always install from the git repo (step 3) until this project ships a
release.

## 1. Get the Cix NPU runtime + model bundle

These are Cix-proprietary/vendored artifacts (the NPU userspace driver
comes from Cix, not from this project) and aren't published to PyPI or
this repo. A standalone bundle mirrors what NCZ-OS ships:

```bash
mkdir -p ~/cix-npu-kit && cd ~/cix-npu-kit
for f in cix-noe-umd_2.0.2_arm64.deb \
         bge-small-zh-v1.5_256.cix \
         bge-small-zh-v1.5-q8_0.gguf \
         bge-small-zh-v1.5-tokenizer.tgz \
         MODELS-README.md; do
  curl -fsSLO "https://pub-d7b784e01679403d9c70fcd23fff5b96.r2.dev/embedkit/$f"
done
tar xzf bge-small-zh-v1.5-tokenizer.tgz
```

| File | What it is |
|---|---|
| `cix-noe-umd_2.0.2_arm64.deb` | Cix's NPU userspace runtime — `libnoe.so.0.6.0` + `libnoe`/`NOE_Engine` Python wheels (cp311/cp312). The only UMD version validated against the in-tree `armchina_npu` KMD; other UMD versions (1.1.1, 3.1.2) fail job-submit. |
| `bge-small-zh-v1.5_256.cix` | NPU model — Cix Compass NN AOT-compiled, INT8, 512-dim embeddings, 256-token max. |
| `bge-small-zh-v1.5-q8_0.gguf` | CPU/Vulkan fallback (`cpu-llamacpp` / Mali GPU) — same embedding space, Q8 GGUF. |
| `bge-small-zh-v1.5-tokenizer.tgz` | Offline HF BERT WordPiece tokenizer, shared by both models. |

## 2. Extract the runtime (do not `dpkg -i`)

The deb's postinst pip-installs into the *system* Python, which fails on
any host running Python ≥3.13. Extract the payload directly instead:

```bash
sudo dpkg -x cix-noe-umd_2.0.2_arm64.deb /
echo "/usr/share/cix/lib" | sudo tee /etc/ld.so.conf.d/cix-noe.conf
sudo ldconfig
```

This lands `libnoe.so{,.0,.0.6.0}` under `/usr/share/cix/lib` and the wheels
under `/usr/share/cix/pypi/`.

## 3. Build a venv and install

Check what actually got extracted before writing the pip command — wheel
filenames vary by UMD version and are NOT always cp311/cp312-tagged:

```bash
ls /usr/share/cix/pypi/
# e.g. libnoe-2.0.0-py3-none-manylinux2014_aarch64.whl
#      NOE_Engine-2.0.0-py3-none-manylinux2014_aarch64.whl
```

```bash
python3.11 -m venv ~/embed-venv
source ~/embed-venv/bin/activate

pip install /usr/share/cix/pypi/libnoe-*.whl \
            /usr/share/cix/pypi/NOE_Engine-*.whl

# Not on PyPI (see warning above) — install from git, with the cix extra:
pip install "mnemos-embedkit[all-arm-cix] @ git+https://gitlab.com/ncz-os/mnemos-embedkit.git"

python -c "import libnoe; print('NPU binding OK:', libnoe.__file__)"
```

If that import fails, `cix-npu` will not be available — check the venv's
Python version (`python --version`) before anything else.

Working from a local clone instead of the git URL: `git clone
https://gitlab.com/ncz-os/mnemos-embedkit.git && cd mnemos-embedkit && pip
install -e ".[all-arm-cix]"`.

## 4. Stage the models

```bash
mkdir -p ~/embed-models
cp ~/cix-npu-kit/bge-small-zh-v1.5_256.cix   ~/embed-models/
cp ~/cix-npu-kit/bge-small-zh-v1.5-q8_0.gguf ~/embed-models/
cp -r ~/cix-npu-kit/bge-small-zh-v1.5-tokenizer ~/embed-models/bge-small-zh-v1.5
```

## 5. Verify + use

```python
import embedkit

for a in embedkit.Engine.list_adapters():
    print(f"{a['tier']:3s} {a['name']:18s} {a['available']!s:5s} {a['reason']}")
# npu cix-npu             True  ...

eng = embedkit.Engine.auto()   # picks cix-npu when libnoe + /dev/aipu are present
vec = eng.embed("hello world")
print(eng.info())
```

## Using with MNEMOS

```bash
docker run -p 5002:5002 -v mnemos-data:/data ghcr.io/ncz-os/mnemos:latest
```

Point MNEMOS at the venv above (or install `mnemos-embedkit` into whatever
environment MNEMOS runs in) — it calls `embedkit.Engine.auto()` on ingest
with no manual embedding step or per-model wiring.

## Troubleshooting

- **`AttributeError: module 'embedkit' has no attribute 'Engine'`**: you
  installed the *wrong* `embedkit` — a same-named, unrelated package from
  PyPI, not this project. `pip uninstall embedkit`, then install from git
  per step 3.
- **`pip install embedkit[all-arm-cix]` warns `does not provide the extra`
  and installs anyway**: same cause — that's the unrelated PyPI package,
  which has no such extra. It installs a stripped base package instead of
  erroring. Uninstall it; the real extras only exist in this repo.
- **`cix-npu` shows `available=False`**: check the venv's Python version
  and confirm the `libnoe`/`NOE_Engine` wheels actually installed
  (`pip show libnoe`) — don't assume a cp3xx tag requirement without
  checking your actual wheel filenames (see step 3).
- **`import libnoe` fails with a missing `.so`**: `ldconfig` didn't pick up
  `/usr/share/cix/lib` — re-check step 2, and confirm
  `/etc/ld.so.conf.d/cix-noe.conf` exists and `ldconfig` ran without error.
- **NPU job-submit fails / driver-level errors**: this is a kernel driver
  issue, not embedkit — confirm `/dev/aipu` exists and the KMD matches UMD
  2.0.2 (other UMD versions are not validated against the in-tree
  `armchina_npu` driver).
