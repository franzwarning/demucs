# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from typing import Optional
import torch
from cog import BaseModel, BasePredictor, Input, Path

from demucs.apply import apply_model
from demucs.audio import save_audio
from demucs.pretrained import get_model
from demucs.separate import load_track

STEMS = ["vocals", "bass", "drums", "guitar", "piano", "other"]

MODELS = [
    "htdemucs",
    "htdemucs_ft",
    "htdemucs_6s",
    "hdemucs_mmi",
    "mdx",
    "mdx_q",
    "mdx_extra",
    "mdx_extra_q",
]


class ModelOutput(BaseModel):
    vocals: Optional[Path]
    bass: Optional[Path]
    drums: Optional[Path]
    guitar: Optional[Path]
    piano: Optional[Path]
    other: Optional[Path]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.models = {k: get_model(k) for k in MODELS}

    def predict(
        self,
        audio: Path = Input(description="Input audio"),
        model_name: str = Input(
            description="Choose a model", choices=MODELS, default="htdemucs"
        ),
        stem: str = Input(
            default=None,
            choices=STEMS,
            description="Only separate audio into the chosen stem and others (no_stem). ",
        ),
        clip_mode: str = Input(
            description="Strategy for avoiding clipping: rescaling entire signal "
            "if necessary  (rescale) or hard clipping (clamp).",
            choices=["rescale", "clamp"],
            default="rescale",
        ),
        shifts: int = Input(
            description="Number of random shifts for equivariant stabilization."
            "Increase separation time but improves quality for Demucs. 10 was used "
            "in the original paper",
            default=1,
        ),
        overlap: float = Input(default=0.25, description="Overlap between the splits."),
        mp3_bitrate: int = Input(
            description="Bitrate of converted mp3",
            default=320,
        ),
        float32: bool = Input(
            description="Save wav output as float32 (2x bigger).",
            default=False,
        ),
        output_format: str = Input(
            default="mp3",
            choices=["mp3", "wav", "flac"],
            description="Choose the output format",
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        model = self.models[model_name]
        if stem is not None:
            assert (
                stem in model.sources
            ), f"stem {stem} is not in selected model. Supported stems in {model_name} are {model.sources}"

        wav = load_track(str(audio), model.audio_channels, model.samplerate)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        sources = apply_model(
            model,
            wav[None],
            device="cuda",
            split=True,
            shifts=shifts,
            overlap=overlap,
            progress=True,
        )[0]
        sources = sources * ref.std() + ref.mean()

        kwargs = {
            "samplerate": model.samplerate,
            "bitrate": mp3_bitrate,
            "clip": clip_mode,
            "as_float": float32,
            "bits_per_sample": 24,
        }

        output = {k: None for k in STEMS}
        if stem is None:
            for source, name in zip(sources, model.sources):
                out = f"/tmp/{name}.{output_format}"
                save_audio(source.cpu(), out, **kwargs)
                output[name] = Path(out)
        else:
            sources = list(sources)
            out_stem = f"/tmp/{stem}.{output_format}"
            save_audio(sources[model.sources.index(stem)].cpu(), out_stem, **kwargs)
            output[stem] = Path(out_stem)

            sources.pop(model.sources.index(stem))
            # Warning : after poping the stem, selected stem is no longer in the list 'sources'
            other_stem = torch.zeros_like(sources[0])
            for i in sources:
                other_stem += i
            out_no_stem = f"/tmp/others.{output_format}"
            save_audio(other_stem.cpu(), out_no_stem, **kwargs)
            output["other"] = Path(out_no_stem)

        return ModelOutput(
            vocals=output["vocals"],
            bass=output["bass"],
            drums=output["drums"],
            guitar=output["guitar"],
            piano=output["piano"],
            other=output["other"],
        )
