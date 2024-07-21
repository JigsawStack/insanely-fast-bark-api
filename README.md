# Insanely Fast Bark API

An API to generate highly realistic human multilingual speech/audio with [Suno AI's Bark](https://github.com/suno-ai/bark)! Powered by 🤗 Transformers, Optimum & flash-attn

Features:
* 🗣️ Text to Speech/Audio at blazing fast speeds
* 📖 Fully open source and deployable on any GPU cloud provider
* 🗣️ Built-in long audio generation support with batching
* ⚡ Easy to use API
* 📃 Async background tasks and webhooks
* 🔥 Optimized for concurrency and parallel processing
* 🔒 Authentication for secure API access
<!-- * 🧩 Fully managed API available on [JigsawStack](https://jigsawstack.com/text-to-speech) -->

This project is focused on providing a deployable blazing fast, near-realtime bark API with truss on Baseten cloud infra by using cheaper GPUs and less resources. 

With [Baseten](https://www.baseten.co/), I've set up the `config.yaml` file to easily deploy on their infra!

Here are some benchmarks we ran on Nvidia A10G - 16GB and Baseten GPU infra👇
| Optimization type    | Time to Transcribe (1 min of Audio) |
|------------------|------------------|
| **bark-small (Transformers) (`fp32`)** | **~2 (*1 min 30 sec*)** |
| **bark-small (Transformers) (`fp32` + `baseten startup`)** | **~2 (*2 min 10 sec*)** |
| **bark-small (Transformers) (`fp16` + `Flash Attention 2`)** | **~0.3 (*18 sec*)**            |
| **bark-small (Transformers) (`fp16` + `Flash Attention 2` + `baseten startup`)** | **~0.3 (*38 sec*)**            |
| **bark-small (Transformers) (`fp16` + `batching (long audio)` + `Flash Attention 2`)** | **~0.3 (*20 sec*)**|
| **bark-small (Transformers) (`fp16` + `batching (long audio)` + `Flash Attention 2` + `baseten startup`)** | **~0.3 (*40 sec*)**|

The estimated startup time for the Baseten machine with GPU and loading up the model is around ~20 seconds. The rest of the time is spent on the actual computation.

## Deploying to Baseten
- Follow the [setup guide](https://docs.baseten.co/quickstart#setup) to get Truss CLI installed and authenticated with Baseten API key
- Clone the project locally and open a terminal in the root
- run `truss push --publish --trusted` to deploy the model

Your API should look something like this:

```
https://{model_id}.api.baseten.co/production/predict
```

## Deploying to other cloud providers
Many of the optimization on Bark is based on [this article](https://huggingface.co/blog/optimizing-bark) that runs on transformers. All the code can be found in the `model.py` file which can be hosted on any other GPU providers using docker or other methods.

<!-- ## Fully managed and scalable API 
[JigsawStack](https://jigsawstack.com) provides a bunch of powerful APIs for various use cases while keeping costs low. This project is available as a fully managed API [here](https://jigsawstack.com/text-to-speech) with enhanced cloud scalability for cost efficiency and high uptime. Sign up [here](https://jigsawstack.com) for free! -->
### Endpoints
Learn how to can your model from the API [here](https://docs.baseten.co/invoke/quickstart)

#### Base URL
```
POST https://{model_id}.api.baseten.co/production/predict
```

Generate audio/speech from text
##### Body params (JSON)
| Name    | value |
|------------------|------------------|
| prompt (Required) | Text of speech you would like to speak out or prompt of audio. [Learn more](https://github.com/suno-ai/bark) |
| voice | [List of voices](https://huggingface.co/suno/bark/tree/main/speaker_embeddings) default: `v2/en_speaker_1` |
| do_sample | Sample voice from voice field default: `true` |
| fine_temperature | Value between `0 - 1` default: None |
| coarse_temperature | Value between `0 - 1` default: None |
| batch_chunk_size | Words per sentence of audio to generate, a single sentence should be less than ~13s which is around 17 words default: `17` |

#### Webhook URL
```
POST https://{model_id}.api.baseten.co/production/async_predict
```

##### Body params (JSON)
| Name    | value |
|------------------|------------------|
| model_input (Required) |  body params of above API |
| webhook_endpoint | callback url |


## Notes

The model uses `suno/bark-small` which offers additional speed-up with the trade-off of slightly lower quality compared to `suno/bark`.
- If you want to switch over to the larger version of bark, you can replaces all references of `suno/bark-small` with `suno/bark` in both `config.yaml` and `model.py` files.

Bark is a highly experimental model and is still in the research phase. The quality of the audio generated may not be perfect and may contain artifacts and noise.
- Bark works great with shorter sentences around 13s (~17 words) of audio for reduced artifacts and noise. Longer audio may contain more blank noise and spaces between sentences.

## Acknowledgements

1. [Yoach Lacombe](https://huggingface.co/ylacombe) for writing a great tutorial on how to optimize bark with transformers and flash.
2. [SunoAI](https://suno.com/) 


<!-- ## JigsawStack
This project is part of [JigsawStack](https://jigsawstack.com) - A suite of powerful and developer friendly AI APIs for various use cases while keeping costs low. Sign up [here](https://jigsawstack.com) for free! -->