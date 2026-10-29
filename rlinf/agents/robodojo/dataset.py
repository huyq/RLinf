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
"""Saved-layout inputs for reproducible RoboDojo GRPO groups."""

import torch
from torch.utils.data import Dataset

from rlinf.data.datasets.common.item import DatasetItem


class RoboDojoDataset(Dataset):
    """Repeat saved layout IDs; metadata, rather than shuffled sample index, owns the seed."""

    def __init__(self, seeds: list[int], repeat: int, tokenizer) -> None:
        self.seeds = list(seeds)
        self.repeat = repeat
        self.prompt = torch.tensor(
            tokenizer.encode(
                "Control the robot using its cameras.", add_special_tokens=False
            ),
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.seeds) * self.repeat

    def __getitem__(self, index: int) -> DatasetItem:
        seed = self.seeds[index % len(self.seeds)]
        return DatasetItem(
            prompt=self.prompt.clone(),
            length=len(self.prompt),
            answer={"seed": seed},
            idx=index,
        )
