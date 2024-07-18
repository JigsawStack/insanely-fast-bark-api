import torch
from transformers import BarkModel
from transformers import AutoProcessor
import scipy
import base64
import io
import numpy as np


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None
        self.processor = None

    def load(self):

        model = BarkModel.from_pretrained(
            "suno/bark-small",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        ).to("cuda:0")

        model.enable_cpu_offload()

        self._model = model
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")

    def predict(self, request: dict):
        prompt = request.get("prompt")
        voice = request.get("voice") or "v2/en_speaker_1"
        inputs = self.processor(prompt, voice_preset=voice).to("cuda:0")
        output = self._model.generate(**inputs)

        #  do_sample=True,
        # fine_temperature=0.4,
        # coarse_temperature=0.8,

        audio_array = output.cpu().numpy().squeeze()
        # https://github.com/suno-ai/bark/issues/478#issuecomment-1858425847, float16 requires adjustment to the audio array which is float32 by default
        audio_array /= 1.414
        audio_array *= 32767
        audio_array = audio_array.astype(np.int16)

        sample_rate = self._model.generation_config.sample_rate
        audio = arr_to_b64(audio_array, sample_rate)
        return {
            "data": audio,
        }


def arr_to_b64(arr, sample_rate):
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    scipy.io.wavfile.write(byte_io, sample_rate, arr)
    wav_bytes = byte_io.read()
    audio_data = base64.b64encode(wav_bytes).decode("UTF-8")
    return audio_data
