#!/bin/bash
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script automated the process to release the component images.
# To run it, find a good release candidate commit SHA from ml-pipeline-staging project,
# and provide a full github COMMIT SHA to the script. E.g.
# ./release.sh 2118baf752d3d30a8e43141165e13573b20d85b8
# The script copies the images from staging to prod, and update the local code.
# You can then send a PR using your local branch.

set -xe

images=(
  "ml-pipeline-dataflow-tf-predict"
  "ml-pipeline-dataflow-tfdv"
  "ml-pipeline-dataflow-tft"
  "ml-pipeline-dataflow-tfma"
)

COMMIT_SHA=$1
FROM_GCR_PREFIX='gcr.io/staging/sdai-pipeline-components/'
TO_GCR_PREFIX='gcr.io/release/sdai-pipeline-components/'
REPO=SmartDeployAI/chicago-taxi-cab
PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

if [[ -z "$COMMIT_SHA" ]]; then
  echo "Usage: release.sh <commit-SHA>" >&2
  exit 1
fi

# Checking out the repo
clone_dir=$(mktemp -d)
echo clone_dir
git clone "git@github.com:${REPO}.git" "$clone_dir"
cd "$clone_dir"
branch="release-$COMMIT_SHA"
# Creating the release branch from the specified commit
release_head=master
git checkout "$release_head" -b "$branch"

# Releasing the container images to public and updating components.
for image in "${images[@]}"
do
    TARGET_IMAGE_BASE=${TO_GCR_PREFIX}${image}
    TARGET_IMAGE=${TARGET_IMAGE_BASE}:${COMMIT_SHA}

    # Move image from test to prod GCR
    gcloud container images add-tag --quiet \
    ${FROM_GCR_PREFIX}${image}:${COMMIT_SHA} ${TARGET_IMAGE}

    # Update the code
    find components samples -type f | while read file; do sed -i -e "s|${TARGET_IMAGE_BASE}:\([a-zA-Z0-9_.-]\)\+|${TARGET_IMAGE}|g" "$file"; done
done

