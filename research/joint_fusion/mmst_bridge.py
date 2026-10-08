"""Load the pinned author architecture without editing the archived source."""
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
REVISION = "615666c8d9fcd704acb662c13065703cbf2eab70"
SOURCE = ROOT / "research/joint_fusion/author_protocol/full_source_615666c8" / ("MMST-ViT-" + REVISION)


def enable_dependencies():
    for directory in (ROOT / ".deps", ROOT / ".deps-author", SOURCE):
        if directory.is_dir() and str(directory) not in sys.path:
            sys.path.insert(0, str(directory))


def make_model(config):
    enable_dependencies()
    import torch
    from torch.utils.checkpoint import checkpoint
    from models_pvt_simclr import PVTSimCLR
    from models_mmst_vit import MMST_ViT

    pvt = PVTSimCLR("pvt_tiny", out_dim=config["embed_dim"], context_dim=9, pretrained=False)

    def forward_pvt(self, x, context=None):
        # The upstream .squeeze() removes the batch axis for a final chunk of
        # length one. Preserve B,N,D; this changes no weights or layer shapes.
        h = self.backbone.forward_features(x)
        z = self.proj(h)
        context = self.proj_context(self.norm1(context))
        return self.mm_transformer(z, context=context)[:, 0]

    pvt.forward = types.MethodType(forward_pvt, pvt)
    model = MMST_ViT(out_dim=2, pvt_backbone=pvt, context_dim=9,
                     dim=config["embed_dim"], num_grid=config["grid_token_capacity"],
                     batch_size=config["image_chunk_size"])

    def features(self, x, ys):
        x = x.reshape(-1, *x.shape[-3:])
        ys = ys.reshape(-1, *ys.shape[-2:])
        result = []
        for start in range(0, len(x), self.batch_size):
            a, b = x[start:start+self.batch_size], ys[start:start+self.batch_size]
            if self.training and torch.is_grad_enabled():
                z = checkpoint(self.pvt_backbone, a, b, use_reentrant=False)
            else:
                z = self.pvt_backbone(a, context=b)
            result.append(z)
        return torch.cat(result)

    # Checkpoint activations so all spatial grids remain in the sample on a
    # laptop GPU. This is not freezing the image backbone or feature caching.
    model.forward_features = types.MethodType(features, model)
    return model
