# Running Llama4 Scout-17B with FP8 Quantization on 2 Cards Using Intel® Gaudi® 3 via vLLM (Offline Serving)

Running inference with Llama4 Scout using FP8 quantization on Gaudi3.

### pull the docker images with help of below command:
```
docker pull vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest
docker run -idt --runtime=habana -e HABANA_VISIBLE_DEVICES=all --cap-add=sys_nice --net=host --ipc=host <image>
docker exec -it <container_id> /bin/bash
```

### Download Scout-17B llama4

```
cd ~/
mkdir models && cd models
export MODEL=meta-llama/Llama-4-Scout-17B-16E-Instruct
export HF_TOKEN=<huggingface_token>
huggingface-cli download --local-dir Llama-4-Scout-17B-16E-Instruct ${MODEL} --token ${HF_TOKEN}
```


### Cloning the vLLM repository and installing dependencies
```
cd ~/

git clone https://github.com/HabanaAI/vllm-fork -b llama4 && cd vllm-fork
pip install -r requirements-hpu.txt; VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation;

> Note: if you're using 1.20.0, you might need to `pip install numpy==1.26.4" if it's not this version yet

> Note: this is from requirements-hpu.txt, for some reason something is uninstalling our version
pip install git+https://github.com/HabanaAI/vllm-hpu-extension.git@145c63d

# install dependencies for llama4
pip install pydantic msgspec cachetools cloudpickle psutil zmq blake3 py-cpuinfo aiohttp openai uvloop fastapi uvicorn watchfiles partial_json_parser python-multipart gguf llguidance prometheus_client numba compressed_tensors datasets
```

### To run FP8 Scout-17B llama4 model on 2 cards, please follow the below steps:

Review the tensor_parallel size setting in the script, as it controls which quantization files are generated and on how many cards they are saved.

Update test_measure.py to set tensor_parallel size. For example, since I want to run on 2 cards, I changed tensor_parallel from 8 to 2.

<img src="https://github.com/user-attachments/assets/1c6dc2b4-c77a-4b50-94ca-31dc8ecced9f" width="700"/>


### Quantization setup
```
# install specific INC
pip uninstall -y neural-compressor neural-compressor-pt
git clone -b dev/llama4_launch https://github.com/intel/neural-compressor.git
cd neural-compressor
pip install -e .
```

Once you complete the below step, a quant.json file and FP8 metadata will be generated in the nc_workspace (Neural Compressor) folder. These files are then used for inference.
```
cd ~/vllm-fork/llama4-scripts

# calibration
QUANT_CONFIG=measure.json PT_HPU_LAZY_MODE=1 python test_measure.py --model_id ~/models/Llama-4-Scout-17B-16E-Instruct/
```
After running the above command and setting tensor_parallel to 2, the output appears as shown below.
<img src="https://github.com/user-attachments/assets/c8cdc0c7-42e6-4514-b77d-7ba53d1beb83" width="900"/>

### Quantization Test

Update test_vllm_quant.py to set tensor_parallel size. For example, since I want to run on 2 cards, I changed tensor_parallel from 8 to 2.
```
# quantization
QUANT_CONFIG=quant.json PT_HPU_LAZY_MODE=1 python test_vllm_quant.py --model_id ~/models/Llama-4-Scout-17B-16E-Instruct/
```
After submitting the image, I received the following output

Input Image:

<img src="https://github.com/user-attachments/assets/8d8b7da3-08d7-41e9-b55f-77657b372545" width="500"/>

Output:

<img src="https://github.com/user-attachments/assets/4ba57fb9-e62c-4d7e-a237-08b3aafeb9d0" width="900"/>


Memory Usage:

<img src="https://github.com/user-attachments/assets/29e9f175-6466-4e1c-bf2f-428d01d9b4c0" width="500"/>







