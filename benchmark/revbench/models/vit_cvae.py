"""A transformer encoder for SubCellSpace, as an ablation of the shipped CVAE.

The shipped model encodes a 100x100x2 image (blurred gene render + graded DAPI)
with four convolution blocks and two fully-connected layers -- **3,727,770
parameters** -- down to a 15-dimensional latent. This module replaces *only that
encoder* with a vision transformer and changes nothing else: same latent
dimension, same beta, same learning rates, same decoder reconstructing
100x100x2, same two ``sample_classifier`` heads, same training split.

Why a transformer might help here
---------------------------------
The six simulated classes are defined by the *relationship* between the
transcript cloud and two boundaries -- the nucleus and the cell membrane --
which can sit far apart in the image. A convolution stack only reaches a global
receptive field after four poolings, by which point it is down to 4x4.
Self-attention relates any patch to any other in the first layer. That is the
hypothesis; a null result is equally reportable, and the held-out loss curve
recorded in the checkpoint is the evidence for why.

Why subclassing works
---------------------
``CVAE_hard_coded.train_model_with_validation_with_2_classifiers`` never names
``encoder_conv`` or ``encoder_FC``. It uses only ``self.embed(x)``,
``self.forward(x)``, ``self.encoder_layers`` (to toggle train/eval while the
classifier heads are fitted) and the two heads. So replacing the encoder and
``embed`` inherits the whole training loop, the beta-VAE loss, the classifier
schedule and -- importantly -- the ``losses`` dict whose ``reconstruction_loss``
and ``val_loss`` are what ``ablations.training_curves`` and
``generalisation_table`` read.

Two variants
------------
``small``  A ViT sized to *this* data: patch 10 on the native 100x100, so no
           upsampling and no token straddling the nucleus boundary at 7px
           granularity. Matched to the conv encoder's parameter count, which is
           what makes it an architecture ablation rather than a capacity change.
``b16``    ``google/vit-base-patch16-224-in21k`` fine-tuned, 85,798,656 params,
           23x the encoder it replaces. Needs the input upsampled 100->224 and a
           third channel; tests transfer, not architecture, and is only readable
           against ``small``.

The third channel for ``b16`` is a **zero plane, not a copy of the gene
channel**: a duplicate silently doubles that channel's weight through the patch
projection. Inputs are rescaled [0,1] -> [-1,1] to match the ViT preprocessing
the checkpoint was trained with (``image_mean`` and ``image_std`` are both 0.5).

**No augmentation is applied anywhere.** ``01_simulated`` scores
``rotation_interval_45`` and ``n_spots_interval_fixed`` as negative controls;
rotation augmentation would force rotation invariance and make the variant look
good on a control that is meant to be measuring the encoder rather than the
augmentation.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Everything the shipped model fixes. An ablation is only interpretable if the
#: reader can see that the encoder is the only thing that moved.
FIXED = {"latent_dim": 15, "n_channels": 2, "img_shape": (100, 100),
         "beta": 1e-4, "lr": 1e-5, "class_lr": 1e-3, "n_classes": 6,
         "batch_size": 512, "reduction": "sum", "seed": 101}


class ViTTrunk(nn.Module):
    """A plain pre-norm ViT returning the CLS token after the final LayerNorm.

    Deliberately hand-rolled rather than pulled from ``timm``/``transformers``:
    at patch 10 on 100x100 there is no pretrained model to reuse, the variant
    must run in whichever environment the encode script uses, and a from-scratch
    trunk keeps the comparison about architecture rather than about somebody
    else's initialisation.

    ``enable_nested_tensor=False`` because the nested-tensor fast path activates
    only in eval and only for some shapes, so leaving it on would make training
    and inference take numerically different routes through the same weights.
    """

    def __init__(self, in_ch: int = 2, img: int = 100, patch: int = 10,
                 width: int = 224, depth: int = 6, heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        if img % patch:
            raise ValueError("patch %d does not tile %dpx" % (patch, img))
        self.grid = img // patch
        n_tokens = self.grid ** 2
        self.width = width
        self.proj = nn.Conv2d(in_ch, width, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=int(width * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth,
                                            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)          # (B, n_tokens, width)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos
        return self.norm(self.blocks(x))[:, 0]               # CLS


class HFViTTrunk(nn.Module):
    """``transformers.ViTModel`` returning the CLS token after its final LayerNorm.

    ``add_pooling_layer=False`` on purpose: the pooler is a tanh projection that
    the ``small`` variant has no counterpart for, and the ``Linear(width, 500)``
    that follows in ``ViTCVAE`` plays that role for both variants. Using it here
    would make the two encoders differ by one more thing than the architecture.
    """

    def __init__(self, name: str = "google/vit-base-patch16-224-in21k",
                 pretrained: bool = True):
        super().__init__()
        from transformers import ViTConfig, ViTModel
        if pretrained:
            self.vit = ViTModel.from_pretrained(name, add_pooling_layer=False)
        else:                                    # for a rebuild from a state dict
            self.vit = ViTModel(ViTConfig.from_pretrained(name),
                                add_pooling_layer=False)
        self.width = int(self.vit.config.hidden_size)

    def forward(self, x):
        return self.vit(pixel_values=x).last_hidden_state[:, 0]


#: variant -> how the trunk is built and how a 100x100x2 batch reaches it.
VARIANTS: Dict[str, Dict[str, Any]] = {
    "small": {
        "encoder_kind": "vit_scratch_p10",
        "trunk": dict(in_ch=2, img=100, patch=10, width=224, depth=6, heads=8,
                      mlp_ratio=4.0),
        "input_spec": ("native 100x100x2, no resize, no third channel, "
                       "per-image min-max as the shipped model"),
        "pretrained": None,
    },
    "b16": {
        "encoder_kind": "vit_base_patch16_224_in21k",
        "hf_name": "google/vit-base-patch16-224-in21k",
        "input_spec": ("resize 100->224 bicubic, third channel a ZERO plane "
                       "(not a copy of the gene channel), rescaled [0,1]->[-1,1] "
                       "to match the checkpoint's image_mean/std of 0.5"),
        "pretrained": "google/vit-base-patch16-224-in21k",
    },
}


class ViTCVAE(nn.Module):
    """The shipped CVAE with its convolutional encoder replaced by a ViT.

    Not a ``CVAE_hard_coded`` subclass by inheritance alone -- it *is* one, but
    the base class is imported lazily so this module can be read (and its specs
    inspected) without the repo-root ``CVAE`` module on ``sys.path``. Use
    :func:`build_vit_cvae`, which returns a real subclass instance.
    """


def _base_class():
    """``CVAE_hard_coded``, imported from wherever the caller put it on the path."""
    try:
        from CVAE import CVAE_hard_coded
    except ModuleNotFoundError as e:                      # pragma: no cover
        raise ModuleNotFoundError(
            "ViTCVAE subclasses CVAE_hard_coded, so the repo root holding "
            "CVAE.py must be on sys.path before building one (the same "
            "requirement cvae_encode.load_model documents for the shipped "
            "checkpoints)") from e
    return CVAE_hard_coded


def build_vit_cvae(variant: str = "small", pretrained: bool = True,
                   latent_dim: int = FIXED["latent_dim"], **trunk_kw):
    """A ViT-encoder CVAE. Everything but the encoder is the shipped model's.

    ``pretrained=False`` builds ``b16``'s architecture with random weights, which
    is what a rebuild-from-state-dict needs; training always uses True.
    """
    if variant not in VARIANTS:
        raise ValueError("variant must be one of %s, got %r"
                         % (sorted(VARIANTS), variant))
    spec = VARIANTS[variant]
    Base = _base_class()

    class _ViTCVAE(Base):
        """Overrides the encoder and ``embed``; inherits decoder, loss, trainer."""

        def __init__(self):
            super().__init__(img_shape=FIXED["img_shape"],
                             n_channels=FIXED["n_channels"],
                             latent_dim=latent_dim, kernel_size=3)
            # the conv encoder is what is being ablated away; the decoder that
            # `super().__init__` built is kept exactly as it is, so the
            # reconstruction target stays 100x100x2 and `reconstruction_loss`
            # and `val_loss` remain on the reference model's scale.
            del self.encoder_conv
            del self.encoder_FC

            if variant == "small":
                kw = dict(spec["trunk"]); kw.update(trunk_kw)
                self.encoder_vit = ViTTrunk(**kw)
            else:
                self.encoder_vit = HFViTTrunk(spec["hf_name"], pretrained=pretrained)

            self.encoder_FC = nn.Sequential(
                nn.Linear(self.encoder_vit.width, 500), nn.ReLU(inplace=True))
            # rebound because enc_mu/enc_logvar are the base's, and the trainer
            # toggles train/eval over exactly this list
            self.encoder_layers = [self.encoder_vit, self.encoder_FC,
                                   self.enc_logvar, self.enc_mu]
            self.variant = variant
            self.encoder_kind = spec["encoder_kind"]
            self.input_spec = spec["input_spec"]
            #: bfloat16 autocast confined to the trunk. Set by the trainer; see
            #: `embed`. Off by default, so the default is the reference's fp32.
            self.amp = False

        def to_encoder_input(self, x):
            """(B,2,100,100) -> whatever this variant's trunk expects."""
            if self.variant == "small":
                return x
            x = F.interpolate(x, size=(224, 224), mode="bicubic",
                              align_corners=False).clamp(0.0, 1.0)
            # a ZERO third plane: copying the gene channel would double its
            # weight through the patch projection without adding information
            z = torch.zeros_like(x[:, :1])
            return torch.cat([x, z], dim=1) * 2.0 - 1.0

        def embed(self, x):
            """(z, mu, logvar) -- the same contract the trainer and decoder use.

            When ``self.amp`` is set, bfloat16 autocast is confined to the ViT
            trunk and its output is cast straight back to float32. Confining it
            matters twice over: the trunk is where essentially all of the cost
            is, and the reference trainer's losses are ``nn.BCELoss``, which
            autocast refuses outright -- so a blanket ``torch.autocast`` around
            the training call raises before the first batch finishes. Everything
            outside the trunk therefore runs in fp32 exactly as the reference
            does, including the sum-reduced reconstruction loss.
            """
            xin = self.to_encoder_input(x)
            if self.amp and xin.is_cuda:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feats = self.encoder_vit(xin)
                feats = feats.float()
            else:
                feats = self.encoder_vit(xin)
            h = self.encoder_FC(feats)
            mu, logvar = self.enc_mu(h), self.enc_logvar(h)
            std = torch.exp(logvar / 2)
            return mu + std * torch.randn_like(std), mu, logvar

    _ViTCVAE.__name__ = "ViTCVAE_%s" % variant
    _ViTCVAE.__qualname__ = _ViTCVAE.__name__
    return _ViTCVAE()


def encoder_parameter_count(model) -> int:
    """Parameters in the encoder only -- the number the ablation is matched on."""
    return sum(p.numel() for layer in model.encoder_layers
               for p in layer.parameters())


def describe(model) -> Dict[str, Any]:
    """The provenance stamped into the checkpoint and every dump's meta."""
    return {"encoder_kind": model.encoder_kind, "variant": model.variant,
            "input_spec": model.input_spec,
            "n_channels": FIXED["n_channels"],
            "latent_dim": int(model.enc_mu.out_features),
            "encoder_parameters": encoder_parameter_count(model),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "augmentation": "none (geometric or intensity)"}


def load_vit_cvae(ckpt_path, device: Optional[str] = None):
    """Rebuild a variant from its checkpoint's ``cvae_state_dict``.

    The checkpoints deliberately store **no pickled model object**: ``b16`` holds
    a ``transformers`` module, and ``05_ablations`` runs in ``subcellspace_env``
    where ``transformers`` is absent, so a pickled instance would make
    ``torch.load`` fail there for a notebook that only ever wanted the losses.
    Rebuilding from weights costs nothing and keeps the checkpoint readable
    everywhere.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(str(ckpt_path), map_location="cpu")
    variant = ck.get("variant")
    if variant is None:
        raise ValueError("%s has no 'variant' key -- not a ViT checkpoint"
                         % ckpt_path)
    kw = dict(ck.get("trunk_kw") or {})
    ld = int(ck.get("ld", FIXED["latent_dim"]))
    model = build_vit_cvae(variant, pretrained=False, latent_dim=ld, **kw)
    # `train_model_with_validation_with_2_classifiers` attaches the two heads to
    # the model as it starts, so they are in the saved state dict while a freshly
    # built model has none. Re-attach before loading rather than dropping them:
    # strict=True is worth keeping, and a variant's own head calls are then
    # available to anyone who wants them.
    from CVAE import sample_classifier
    n_classes = int(ck.get("n_classes", FIXED["n_classes"]))
    model.binary_classifier = sample_classifier(n_classes=2, n_input=ld)
    model.classifier = sample_classifier(n_classes=n_classes, n_input=ld)
    model.load_state_dict(ck["cvae_state_dict"], strict=True)
    model.to(dev).eval()
    return model, ck
