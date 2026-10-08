# Sources, attribution and scope

The archived architecture is from Fudong Li and the MMST-ViT contributors:

- https://github.com/fudong03/MMST-ViT
- Revision: `615666c8d9fcd704acb662c13065703cbf2eab70`
- Original license: Creative Commons Attribution-NonCommercial 4.0 International. The complete upstream license is retained without modification under `research/joint_fusion/author_protocol/full_source_615666c8/MMST-ViT-615666c8d9fcd704acb662c13065703cbf2eab70/LICENSE`.
- Related dataset/research repository: https://github.com/fudong03/CropNet
- Dataset: https://huggingface.co/datasets/fudong03/Tiny-CropNet
- Dataset revision: `07260d9d23d7093abd89cb5a78f8d82a439c8816`

The upstream architecture files are preserved. Local adaptations are in `mmst_bridge.py` (batch-dimension handling and activation checkpointing), the data/normalization pipeline, paired training methods, asynchronous aggregation, recovery, and reporting modules. These experiments use random initialization without the original SSL pretraining.

Upstream copyright and license terms remain applicable to their respective material. This publication does not relicense third-party material. No additional general license grant for the new project code is made by this initial sharing release; contact the repository owner for reuse permissions beyond applicable existing licenses.

The included results concern soybean yield, not corn. Public county FIPS identifiers, observed yields, predictions, numerical training traces and archived upstream county lists are included for scientific review. Raw imagery, weather archives and model weights are obtained or generated separately.
