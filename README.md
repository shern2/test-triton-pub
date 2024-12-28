## Pre-requisites

Assumes you have the following setup: Docker, Nvidia Container Toolkit, Nvidia GPU, conda (This project was tested on Ubuntu 22.04).


## Installation

### Setup local python environment

> WARNING Strongly recommend to use a separate python env as `tensorrt` installation / updates can mess files up and it's easier to do a clean install...

```bash
cd $PROJECT_DIR # i.e. the repo root folder
conda create -n compile python=3.12
conda activate compile
pip install -r requirements_compile.txt
```

### Setup triton inference server container

```bash
cd $PROJECT_DIR
docker compose build
# Start the container running idly in the background.
docker compose up -d
```

## Directory structure

```bash
# The FastAPI application serving as a proxy to the triton server, plus some start scripts
app/
# The model repository for Triton
# NOTE: Follows a strict directory structure of `models/<model_name>/<model_version>/`.
# The `<model_name>` MUST agree with the `name` in the `config.pbtxt` file.
models/
# Jupyter notebooks (to run locally for compiling models and testing)
notebooks/
# Helper functions/scripts
src/

```

### case 1: Simple neural network model

```bash
# Run `notebooks/01_compile_pt__simple_net.ipynb.ipynb` to export the simple neural network model and output a TorchScript `model.pt` file under `models/simple_net/1/`.

# Enter the container
docker exec -it triton /bin/bash

# Start the triton server (simple_net model only)
./dev_start.sh

# Use `notebooks/00_test_fastapi_proxy_output.ipynb` to validate that the Triton outputs are as expected.
```

### case 2: FinBERT model

Run `notebooks/02_compile_onnx__finbert.ipynb`. This will
1. Export the FinBERT model to ONNX
1. Optimize the ONNX model using TensorRT and save it to the model repository
1. Save the tokenizer as a "model" in the model repository
1. Note that the prediction pipeline (a.k.a. "ensemble" model in Nvidia lingo) is **pre-setup** in `models/finbert/`.

Modify `dev_start.sh` `tritonserver` command to start the FinBERT models, before running it:
```bash
--load-model=finbert-model \
--load-model=finbert-tokenizer \
--load-model=finbert \
```

Validate using `notebooks/00_test_fastapi_proxy_output.ipynb`.


### case 3: FinBERT model (TensorRT) multiple instances

Modify `dev_start.sh` `tritonserver` command to start the FinBERT models, before running it:
```bash
--load-model=finbert-trt-model \
```

Modify the `models/finbert-trt-model/config.pbtxt` file's `instance_group` by varying the number of instances and seeing the GPU usage via `nvidia-smi`. Validate that the GPU memory does not increase linearly with the number of instances, but rather less - i.e. the weight-sharing is working.
```pbtxt
instance_group [
    {
        kind: KIND_GPU
        count: 4
    }
]
```

## References

https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html
https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md
https://pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/torch_compile_transformers_example.html
https://github.com/triton-inference-server/client/blob/main/src/python/examples/simple_http_aio_infer_client.py
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags