#!/usr/bin/env bash
# Lane G entry: FlashInfer 0.6.18 (ckv_scale_arr), SM90 NoPE MLA, then vllm serve.
# Official glm53-flash-arm64-cu130 ships 0.6.17; without 0.6.18, sparse MLA has
# no prefill backend and completions collapse (lock/handle). DeepSeek hotfixes stay out.
set -u
mkdir -p /root/.cache/vllm/jit/triton /root/.cache/vllm/jit/tilelang \
         /root/.cache/vllm/nccl-flight /root/.cache/vllm/pip

has_nope_mla() {
  python3 -c '
import inspect
from flashinfer.mla import BatchMLAPagedAttentionWrapper
params = inspect.signature(BatchMLAPagedAttentionWrapper.run).parameters
raise SystemExit(0 if "ckv_scale_arr" in params else 1)
' 2>/dev/null
}

if ! has_nope_mla; then
  echo "[glm-entry] installing FlashInfer 0.6.18 (ckv_scale_arr)"
  WHEELDIR=/root/.cache/vllm/wheels
  if [ -f "$WHEELDIR/flashinfer_python-0.6.18.dev20260819-py3-none-any.whl" ] \
     && [ -f "$WHEELDIR/flashinfer_cubin-0.6.18.dev20260819-py3-none-any.whl" ]; then
    python3 -m pip install -q --pre --no-deps \
      "$WHEELDIR"/flashinfer_python-0.6.18.dev20260819-py3-none-any.whl \
      "$WHEELDIR"/flashinfer_cubin-0.6.18.dev20260819-py3-none-any.whl
  else
    python3 -m pip install -q --pre --no-deps \
      flashinfer-python==0.6.18.dev20260819 \
      flashinfer-cubin==0.6.18.dev20260819 \
      --extra-index-url https://flashinfer.ai/whl/nightly/ \
      --cache-dir /root/.cache/vllm/pip
  fi
  # jit-cache SM120 cubins fight the SM90 NoPE path on GB10
  python3 -m pip uninstall -q -y flashinfer-jit-cache || true
  # pin from PyPI only; nightly extra-index must not resolve these
  python3 -m pip install -q nvidia-nccl-cu13==2.30.7 nvidia-cutlass-dsl==4.6.2 \
    --index-url https://pypi.org/simple --cache-dir /root/.cache/vllm/pip
  python3 -c "import flashinfer; print('[glm-entry] flashinfer', flashinfer.__version__)"
fi

python3 /sparkduet-patches/glm53-sm90.py || echo "[patches] glm53-sm90 did not apply; engine boots unpatched" >&2
exec vllm serve "$@"
