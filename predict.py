# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from typing import Optional, Iterator
import torch
import tempfile
from cog import BaseModel, BasePredictor, Input, Path
import subprocess
import sys
import yt_dlp as youtube_dl
from demucs.apply import apply_model
from demucs.audio import save_audio
from demucs.pretrained import get_model
from demucs.separate import load_track
import urllib
# from pydub import AudioSegment

STEMS = ["vocals", "bass", "drums", "guitar", "piano", "other"]



ydl_opts = {
    'format': 'bestaudio/best',
    'noplaylist': True
}

MAX_VIDEO_LENGTH_SECONDS = 10 * 60

# https://www.youtube.com/watch?v=vhAznYpU9Ig&list=PLxA687tYuMWh9sWBuPx_CdtnXGfSR8k_O

def upgrade(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--upgrade"])

class FilenameCollectorPP(youtube_dl.postprocessor.common.PostProcessor):
    def __init__(self):
        super(FilenameCollectorPP, self).__init__(None)
        self.filenames = []

    def run(self, information):
        self.filenames.append(information['filepath'])
        return [], information



class ModelOutput(BaseModel):
    vocals: Optional[Path]
    bass: Optional[Path]
    drums: Optional[Path]
    other: Optional[Path]
    title: Optional[str]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = get_model('mdx_extra_q')
        upgrade('yt-dlp')

    def predict(
        self,
        audio_url: str = Input(description="Audio url", default=None),
        shifts: int = Input(
            description="Number of random shifts for equivariant stabilization."
            "Increase separation time but improves quality for Demucs. 10 was used "
            "in the original paper",
            default=1,
        ),
        overlap: float = Input(default=0.25, description="Overlap between the splits."),
    ) -> Iterator[ModelOutput]:
        """Run a single prediction on the model"""

        ####
        # YOUTUBE DL
        ####        
        ytdl_temp_dir = tempfile.mkdtemp()

        ydl = youtube_dl.YoutubeDL({
            **ydl_opts,
            'outtmpl':f"{ytdl_temp_dir}/%(title)s.%(ext)s",
        })

        info_dict = ydl.extract_info(audio_url, download=False)
        duration = info_dict.get('duration')
        if duration:
            if duration > MAX_VIDEO_LENGTH_SECONDS:
                raise Exception('Audio file needs to be under 10 minutes')
        else:
            # get size of file
            req = urllib.request.Request(audio_url, method='HEAD')
            f = urllib.request.urlopen(req)
            size = f.headers['Content-Length']
            size_mb = int(size) / 1000000
            if size_mb > 50:
                raise Exception('Audio file needs to be smaller than 50 mb')


        video_title = info_dict.get('title', None)
        yield ModelOutput(title=video_title)


        filename_collector = FilenameCollectorPP()
        ydl.add_post_processor(filename_collector)
        ydl.download([audio_url])            

        wav = load_track(Path(filename_collector.filenames[0]), self.model.audio_channels, self.model.samplerate)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        sources = apply_model(
            self.model,
            wav[None],
            device="cuda",
            split=True,
            shifts=shifts,
            overlap=overlap,
            progress=True,
        )[0]
        sources = sources * ref.std() + ref.mean()

        kwargs = {
            "samplerate": self.model.samplerate,
            "bits_per_sample": 16,
        }

        output = {k: None for k in STEMS}
        for source, name in zip(sources, self.model.sources):
            wav_out = f"/tmp/{name}.wav"
            save_audio(source.cpu(), wav_out, **kwargs)
            # mp3_out = f"/tmp/{name}.mp3"
            # AudioSegment.from_wav(wav_out).export(mp3_out, format="mp3", bitrate="320k")

            output[name] = Path(wav_out)

        yield ModelOutput(
            vocals=output["vocals"],
            bass=output["bass"],
            drums=output["drums"],
            other=output["other"],
            title=video_title
        )
