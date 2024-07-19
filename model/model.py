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
        # https://huggingface.co/docs/transformers/en/model_doc/bark#combining-optimization-techniques
        # https://huggingface.co/blog/optimizing-bark
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
        prompt = prompt.replace("\n", " ").strip()
        voice = (
            request.get("voice") or "v2/en_speaker_1"
        )  # https://huggingface.co/suno/bark/tree/main/speaker_embeddings/v2

        do_sample = request.get("do_sample") or None
        fine_temperature = request.get("fine_temperature") or None
        coarse_temperature = request.get("coarse_temperature") or None
        batch_chunk_size = request.get("batch_chunk_size") or 17

        sample_rate = self._model.generation_config.sample_rate

        promptWords = prompt.split()
        # chunk sentences by 17 words (<=~13 of audio) since bark can only generate 13 seconds of audio at a time
        chunksList = list(chunks(promptWords, batch_chunk_size))
        sentences = [" ".join(chunk) for chunk in chunksList]
        print("sentences", sentences)

        inputs = self.processor(sentences, voice_preset=voice).to("cuda:0")
        output = self._model.generate(
            **inputs,
            do_sample=do_sample,
            fine_temperature=fine_temperature,
            coarse_temperature=coarse_temperature
        )

        print("output length", len(output))

        audio_array = []

        for outputChunk in output:
            single_audio = outputChunk.cpu().numpy().squeeze()
            # remove 1 seconds of silence from the end of the audio
            single_audio = single_audio[: -1 * sample_rate]
            audio_array += [single_audio]

        audio_array = np.concatenate(audio_array)

        # https://github.com/suno-ai/bark/issues/478#issuecomment-1858425847, float16 requires adjustment to the audio array which is float32 by default
        audio_array /= 1.414
        audio_array *= 32767
        audio_array = audio_array.astype(np.int16)

        audio = arr_to_b64(audio_array, sample_rate)
        return {
            "data": audio,
            "sentences": sentences,
        }


def arr_to_b64(arr, sample_rate):
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    scipy.io.wavfile.write(byte_io, sample_rate, arr)
    wav_bytes = byte_io.read()
    audio_data = base64.b64encode(wav_bytes).decode("UTF-8")
    return audio_data


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
