# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Export a RoboDojo FSDP checkpoint for HF/SGLang evaluation."""

import argparse

from rlinf.agents.robodojo.checkpoint import export_checkpoint


def main() -> None:
    """Export actor full weights and preserve its tokenizer and image processor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    export_checkpoint(args.weights, args.model_path, args.output_dir)


if __name__ == "__main__":
    main()
