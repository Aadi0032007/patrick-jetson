#!/usr/bin/env bash
# Experimental native Q8 build. Uses existing JetPack CUDA; never installs drivers.
set -euo pipefail
if [[ "$(uname -s)" != Linux || "$(uname -m)" != aarch64 ]]; then
    echo 'Run on the Orin NX itself (Linux aarch64), not Windows/WSL x86_64.' >&2
    exit 1
fi
for command in git cmake nvcc; do
    if ! command -v "$command" >/dev/null; then
        echo "Missing $command. Install build tools/use your JetPack CUDA toolkit first." >&2
        exit 1
    fi
done
prefix="${PATRICK_Q8_BUILD_ROOT:-$HOME/.local/share/patrick-q8}"
source_dir="$prefix/codec.cpp"
revision=63ade8bf2fa3451f9eca7ebaa39f792cc529438e
mkdir -p "$prefix"
if [[ ! -d "$source_dir" ]]; then
    git init "$source_dir"
    git -C "$source_dir" remote add origin https://github.com/mybigday/codec.cpp.git
    git -C "$source_dir" fetch --depth 1 origin "$revision"
    git -C "$source_dir" checkout --detach FETCH_HEAD
elif [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" ]]; then
    echo 'Existing source differs; choose a new PATRICK_Q8_BUILD_ROOT.' >&2
    exit 1
fi
git -C "$source_dir" submodule update --init --recursive --depth 1
build_dir="$prefix/build-orin"
cmake -S "$source_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
    -DCODEC_BUILD_TTS_CLI=ON -DCODEC_TTS_BACKBONE=ON -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build "$build_dir" --target tts-cli --parallel "${PATRICK_BUILD_JOBS:-2}"
test -x "$build_dir/tts-cli"
echo "Experimental runner: $build_dir/tts-cli"
echo 'Test synthesis before integration. The reference T3 backbone is CPU-only; CUDA accelerates the codec.'
