#!/usr/bin/env bash
# Export GC upload artifacts:
#   1) container image: docker save IMAGE | gzip -c > IMAGE.tar.gz
#   2) model weights:   model.tar.gz  (upload separately on Grand Challenge)
#
# Runtime constraints (ISLES'26): T4 GPU, ~7 minutes per case.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="isles26_baseline_nnunet5fold"
export DOCKER_CLI_HINTS=false

log() { printf '> %s\n' "$1"; }

log "(Re)build the image"
export DOCKER_QUIET_BUILD=1
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/do_build.sh"

build_timestamp=$(docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")
formatted_build_info=$(echo "$build_timestamp" | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')

# Stable name for upload + timestamped backup
stable_path="${SCRIPT_DIR}/${DOCKER_IMAGE_TAG}.tar.gz"
ts_path="${SCRIPT_DIR}/${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"

log "Saving image via: docker save ${DOCKER_IMAGE_TAG} | gzip -c > …"
# Official GC format: docker save IMAGE | gzip -c > IMAGE.tar.gz
# https://docs.docker.com/engine/reference/commandline/save/
docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$stable_path"
cp -a "$stable_path" "$ts_path"
log "Saved: $(basename "$stable_path") ($(du -h "$stable_path" | cut -f1))"
log "Also:  $(basename "$ts_path")"

log "Packing model.tar.gz (separate GC Model upload)"
tar -czf "${SCRIPT_DIR}/model.tar.gz" -C "${SCRIPT_DIR}/model" .
log "Saved: model.tar.gz ($(du -h "${SCRIPT_DIR}/model.tar.gz" | cut -f1))"

log "Upload ${DOCKER_IMAGE_TAG}.tar.gz as the container; model.tar.gz as the Model."
log "Runtime target: T4, ≤7 min/case (TTA mirroring OFF in this baseline)."
