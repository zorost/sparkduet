# Lane G on TensorFold: the upstream native server for GLM-5.3-Flash EXL3/TR3 on GB10.
#
# No engine patch is applied. The engine is TensorFold (Apache-2.0,
# https://github.com/ashhart/TensorFold), from one of two pinned sources:
#
#   TF_SOURCE=git (default)  built from upstream main at TF_REF. 81de26f6 reports
#                            1.3.0 but sits after the v1.3.0 tag (d3d0bf11): its
#                            changelog lists, as unreleased, programmatic dependent
#                            launches along GLM's CUDA decode chain, next-turn
#                            cache guesses that carry the reply's reasoning
#                            (#3264) and KV page-retirement bookkeeping (#3852).
#   TF_SOURCE=release        the v1.3.0 linux-aarch64 release archive, checked
#                            against its published sha256 and its SHA256SUMS.
#
# Either way the runtime stage adds what a two-rank GLM pair needs beside the
# binary: NCCL, libibverbs with the mlx5 provider (the engine's RoCE gathers) and
# the CUDA 13 runtime with cuBLASLt (the vision tower). TensorFold's LICENSE,
# NOTICE, THIRD_PARTY_NOTICES.md and LICENSES/ stay in /opt/tensorfold.
#
# Build on a Spark (aarch64), then give the worker the same image:
#   scripts/build-tensorfold-image.sh
# or by hand:
#   docker build -t sparkduet-tensorfold:1.3.0 patches/glm-tensorfold
#   docker build --build-arg TF_SOURCE=release -t sparkduet-tensorfold:1.3.0-release patches/glm-tensorfold
#
# [unverified on chassis] Neither source has been built on a Spark. The CUDA
# base tags were not pulled from the build VM: pin them by digest after the
# first pull on the head.

ARG TF_SOURCE=git
ARG CUDA_RUNTIME=nvidia/cuda:13.0.1-runtime-ubuntu24.04
ARG CUDA_DEVEL=nvidia/cuda:13.0.1-devel-ubuntu24.04

# ---- engine from upstream main at a pinned commit ---------------------------
FROM ${CUDA_DEVEL} AS engine-git
ARG TF_REPO=https://github.com/ashhart/TensorFold.git
ARG TF_REF=81de26f649677aa72ae0034d4af6e010cd947038
# build.zig.zon at TF_REF: minimum_zig_version 0.17.0. Zig comes from PyPI's
# ziglang wheel, pinned by hash.
ARG ZIG_VERSION=0.17.0
ARG ZIG_WHEEL_SHA256=d2804f94f54f7fcf8169f4b212a619ad688a4e2e6592e976f0cf2e58e2cd8af3
# libtfgrammar (structured output, required tool calls) builds from xgrammar v0.2.8.
ARG XGRAMMAR_TAG=v0.2.8
ARG XGRAMMAR_COMMIT=97787376faee5ed8466cfad57c99855e4ce2f6aa
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8
RUN test "$(uname -m)" = aarch64 || { echo "build this image on a Spark (aarch64)" >&2; exit 1; } \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git cmake python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*
RUN printf 'ziglang==%s --hash=sha256:%s\n' "${ZIG_VERSION}" "${ZIG_WHEEL_SHA256}" > /tmp/zig.txt \
 && python3 -m pip install --no-cache-dir --break-system-packages --require-hashes -r /tmp/zig.txt \
 && printf '#!/bin/sh\nexec python3 -m ziglang "$@"\n' > /usr/local/bin/zig \
 && chmod +x /usr/local/bin/zig \
 && test "$(zig version)" = "${ZIG_VERSION}"
RUN git clone -q --depth 1 --branch "${XGRAMMAR_TAG}" --recurse-submodules \
      https://github.com/mlc-ai/xgrammar.git /src/xgrammar \
 && test "$(git -C /src/xgrammar rev-parse HEAD)" = "${XGRAMMAR_COMMIT}"
WORKDIR /src/tensorfold
RUN git init -q . \
 && git remote add origin "${TF_REPO}" \
 && git fetch -q --depth 1 origin "${TF_REF}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${TF_REF}"
# One GPU target (GB10, sm_121), as upstream's RUNBOOK builds it, then the
# grammar library from the development build.
RUN zig build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc --prefix /opt/tf-build --summary failures \
 && zig build grammar -Dxgrammar=/src/xgrammar --prefix /opt/tf-build --summary failures \
 && bin="$(find /opt/tf-build -type f -name tensorfold-native -print -quit)" \
 && lib="$(find /opt/tf-build -type f -name libtfgrammar.so -print -quit)" \
 && test -x "${bin}" && test -f "${lib}" \
 && mkdir -p /opt/tensorfold/bin /opt/tensorfold/lib /opt/tensorfold/LICENSES \
 && cp "${bin}" /opt/tensorfold/bin/ \
 && cp "${lib}" /opt/tensorfold/lib/ \
 && cp LICENSE NOTICE THIRD_PARTY_NOTICES.md /opt/tensorfold/ \
 && cp -r LICENSES/. /opt/tensorfold/LICENSES/ \
 && cp /src/xgrammar/LICENSE /opt/tensorfold/LICENSES/xgrammar-LICENSE \
 && cp /src/xgrammar/NOTICE /opt/tensorfold/LICENSES/xgrammar-NOTICE \
 && printf '%s %s\n' "${TF_REF}" "${TF_REPO}" > /opt/tensorfold/SOURCE \
 && /opt/tensorfold/bin/tensorfold-native --version

# ---- engine from the v1.3.0 release archive --------------------------------
FROM ${CUDA_RUNTIME} AS engine-release
ARG TF_VERSION=1.3.0
ARG TF_SHA256=718467e07007c566153bcddbd60b311709dfa8e6e98fab3c67330e0e3052987c
ARG TF_URL=https://github.com/ashhart/TensorFold/releases/download/v${TF_VERSION}/tensorfold-${TF_VERSION}-linux-aarch64.tar.gz
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN test "$(uname -m)" = aarch64 || { echo "build this image on a Spark (aarch64)" >&2; exit 1; } \
 && apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL --retry 3 -o /tmp/tf.tar.gz "${TF_URL}" \
 && echo "${TF_SHA256}  /tmp/tf.tar.gz" | sha256sum -c - \
 && mkdir -p /opt/tensorfold \
 && tar -xzf /tmp/tf.tar.gz -C /opt/tensorfold --strip-components=1 \
 && rm /tmp/tf.tar.gz \
 && (cd /opt/tensorfold && sha256sum -c SHA256SUMS) \
 && test "$(cat /opt/tensorfold/VERSION)" = "${TF_VERSION}" \
 && printf 'v%s %s\n' "${TF_VERSION}" "${TF_SHA256}" > /opt/tensorfold/SOURCE \
 && /opt/tensorfold/bin/tensorfold-native --version

# ---- the engine the build asked for ----------------------------------------
FROM engine-${TF_SOURCE} AS engine

# ---- runtime ----------------------------------------------------------------
FROM ${CUDA_RUNTIME}
# The house NCCL for TP over the GB10 fabric, the same pin as patches/glm-entry.sh.
ARG NCCL_VERSION=2.30.7
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl python3 python3-pip \
      rdma-core libibverbs1 ibverbs-providers \
 && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --break-system-packages "nvidia-nccl-cu13==${NCCL_VERSION}" \
 && lib="$(dirname "$(find / -path '*/nvidia/nccl/lib/libnccl.so.2' -print -quit 2>/dev/null)")" \
 && test -f "${lib}/libnccl.so.2" \
 && mkdir -p /opt/nccl \
 && ln -s "${lib}" /opt/nccl/lib
COPY --from=engine /opt/tensorfold /opt/tensorfold
ENV PATH=/opt/tensorfold/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/nccl/lib:/opt/tensorfold/lib:/usr/local/cuda/lib64
RUN /opt/tensorfold/bin/tensorfold-native --version && cat /opt/tensorfold/SOURCE
LABEL org.opencontainers.image.title="sparkduet-tensorfold" \
      org.opencontainers.image.description="TensorFold native server for SparkDuet Lane G (G_ENGINE=tensorfold)" \
      sparkduet.nccl.version="${NCCL_VERSION}"
