# Running Llama4 Scout-17B with FP8 Quantization on 2 Cards Using Intel® Gaudi® 2 via vLLM (Online Serving)

Inferencing llama4 Scout with fp8 quantization using gaudi3

pull the docker images with help of below command:
```
docker pull vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
docker run -idt --runtime=habana -e HABANA_VISIBLE_DEVICES=all --cap-add=sys_nice --net=host --ipc=host <image>
docker exec -it <container_id> /bin/bash
```

Download Scout-17B llama4

```
cd ~/
mkdir models && cd models
export MODEL=meta-llama/Llama-4-Scout-17B-16E-Instruct
export HF_TOKEN=<huggingface_token>
huggingface-cli download --local-dir Llama-4-Scout-17B-16E-Instruct ${MODEL} --token ${HF_TOKEN}
```


cloning vLLM llama4 repo

```
cd
git clone https://github.com/HabanaAI/vllm-fork -b llama4
cd vllm-fork/
```

To get a FP8 Scout-17B llama4 model, please follow below steps:
```


```
