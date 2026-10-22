#!/usr/bin/env bash

#===============================================================================
# Copyright 2026 Barcelona Supercomputing Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Mirrors oneDNN's own .github/automation/clang-format.sh, scoped to this
# repo: third_party/ is vendored and not ours to reformat.

set -eu

CLANG_FORMAT=clang-format-18

echo "Checking ${CLANG_FORMAT}"
if ! ${CLANG_FORMAT} --version; then
    echo ${CLANG_FORMAT} is not available or not working correctly.
    exit 1
fi

echo "Starting format check..."

src_regex='.*\.(c|h|cpp|hpp|cxx|hxx)$'

# Treat each argument as a base sha to diff against. No arguments checks the
# whole repo (minus third_party/, which is vendored).
if [[ $# -gt 0 ]]; then
    base_sha=$1
    file_list=$(git diff --name-only "$base_sha" | grep -E "$src_regex" | grep -v '^third_party/' || true)
    echo "Checking: $file_list"
    [[ -n "$file_list" ]] && ${CLANG_FORMAT} -style=file -i $file_list
else
    find "$(pwd)" -type f -regextype posix-egrep -regex "$src_regex" \
        -not -path "$(pwd)/third_party/*" -exec "$CLANG_FORMAT" -style=file -i {} \+
fi

RETURN_CODE=0

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Clang-format check FAILED! The following files must be formatted with ${CLANG_FORMAT}:"
    git diff --name-only
    echo
    echo "Changes required to pass this check:"
    git diff
    echo
    RETURN_CODE=3
else
    echo "Clang-format check PASSED!"
fi

exit ${RETURN_CODE}
