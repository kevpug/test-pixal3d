from typing import *
import os
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


# BRIA's RMBG-2.0, which the pipeline config asks for, is a gated repo: it
# needs a Hugging Face account that has accepted the licence and a token in
# the environment. It is a fine-tune of BiRefNet, so the original -- MIT
# licensed and ungated -- is a drop-in replacement when the gate blocks us.
FALLBACK_MODEL = "ZhengPeng7/BiRefNet"

_ACCESS_MARKERS = (
    'gatedrepoerror', 'gated repo', 'restricted', 'unauthorized',
    '401', 'must be authenticated', 'repositorynotfounderror',
)


def _is_access_error(error: BaseException) -> bool:
    """True for 'this repo is gated or you are not logged in'.

    A genuine load failure -- corrupt download, missing dependency -- must
    still surface, so this only matches the authentication wording.
    """
    text = f"{type(error).__name__}: {error}".lower()
    return any(marker in text for marker in _ACCESS_MARKERS)


class BiRefNet:
    def __init__(self, model_name: str = FALLBACK_MODEL):
        # An explicit override wins over whatever the pipeline config asks for.
        self.model_name = os.environ.get('PIXAL3D_REMBG_MODEL') or model_name
        # Background removal only runs for images that have no alpha channel,
        # so the weights are loaded on first use: feeding an already-masked
        # image should not cost a download. Until then this is where the
        # model will go once it exists.
        self.model = None
        self._device = torch.device('cpu')
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _from_pretrained(model_name: str):
        return AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )

    def _load(self):
        if self.model is not None:
            return self.model
        try:
            model = self._from_pretrained(self.model_name)
        except Exception as error:
            if self.model_name == FALLBACK_MODEL or not _is_access_error(error):
                raise
            print(f"[pixal3d] Background-removal model '{self.model_name}' is gated "
                  f"and this Hugging Face login cannot reach it.")
            print(f"[pixal3d] Falling back to '{FALLBACK_MODEL}' -- same architecture, "
                  f"MIT licensed, no login needed.")
            print(f"[pixal3d] To use the original instead: accept the licence at "
                  f"https://huggingface.co/{self.model_name} and run 'hf auth login'.")
            model = self._from_pretrained(FALLBACK_MODEL)
            self.model_name = FALLBACK_MODEL
        model.eval()
        model.to(self._device)
        self.model = model
        return model

    def to(self, device: str):
        # Recorded even when the model is not loaded yet, so that low-VRAM
        # mode moving it on and off the GPU still lands it in the right place.
        self._device = torch.device(device)
        if self.model is not None:
            self.model.to(self._device)

    def cuda(self):
        self.to('cuda')

    def cpu(self):
        self.to('cpu')

    @property
    def device(self) -> torch.device:
        if self.model is None:
            return self._device
        # next() over an empty parameter list raises StopIteration, which is
        # poorly behaved inside a generator; the recorded device is right
        # in that case anyway.
        first = next(self.model.parameters(), None)
        return self._device if first is None else first.device

    def __call__(self, image: Image.Image) -> Image.Image:
        model = self._load()
        image_size = image.size
        # Follow the model rather than hard-coding "cuda": in low-VRAM mode it
        # is moved on and off the GPU around this call, and a CPU-only machine
        # should get a slow run instead of a device-mismatch error.
        input_images = self.transform_image(image).unsqueeze(0).to(self.device)
        # Prediction
        with torch.no_grad():
            preds = model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
