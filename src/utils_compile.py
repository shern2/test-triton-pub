import logging
from pathlib import Path

from pydantic import BaseModel, validator

logger = logging.getLogger(__name__)

# The default config.pbtxt template (transformer type) for Triton Inference Server
DEFAULT_CONFIG_TEMPLATE = """
name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}
input [
    {{
        name: "input_ids"
        data_type: TYPE_INT32
        dims: [ -1 ]
    }},
    {{
        name: "attention_mask"
        data_type: TYPE_INT32
        dims: [ -1 ]
    }},
    {{
        name: "token_type_ids"
        data_type: TYPE_INT32
        dims: [ -1 ]
    }}
]
output [
    {{
        name: "logits"
        data_type: TYPE_FP32
        dims: [ {logits_dims} ]
    }}
]
instance_group [
    {{
        kind: KIND_GPU
        count: 1
    }}
]
dynamic_batching {{
    preferred_batch_size: {preferred_batch_size}
    max_queue_delay_microseconds: {max_queue_delay_microseconds}
}}
"""


class DefaultConfigInput(BaseModel):
    model_name: str
    max_batch_size: int
    logits_dims: int
    preferred_batch_size: list[int]
    max_queue_delay_microseconds: int


class Output(BaseModel):
    models_dir: Path
    pth_model: Path
    pth_cur_script: Path
    compiled_model_nm: str
    version: str

    @validator("version")
    def version_must_be_float_compatible(cls, v):
        try:
            float(v)
        except ValueError:
            raise ValueError(f"Version (i.e. the parent dir) should be numeric. Got: {v}")
        return v


def get_params(file: str, models_dir: Path = None) -> Output:
    """Get the parameters for the compilation script

    Args:
        file (str): The __file__ variable from the script (i.e. script's path)
        models_dir (Path, optional): The directory where the models are stored, defaults to `./models`.

    Returns:
        The parameters for the compilation script
    """
    models_dir = models_dir or Path("./models")
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Compilation expects this directory to exist: {models_dir} ; Please run the script in the right directory or provide the path `pth_model` explicitly"
        )

    pth_cur_script = Path(file)
    compiled_model_nm = pth_cur_script.parents[1].name
    version = pth_cur_script.parents[0].name
    pth_model = models_dir / f"{compiled_model_nm}/{version}/model.pt"

    pth_model.parent.mkdir(parents=True, exist_ok=True)

    return Output(
        models_dir=models_dir,
        pth_model=pth_model,
        pth_cur_script=pth_cur_script,
        compiled_model_nm=compiled_model_nm,
        version=version,
    )


def check_model_name(model_name: str) -> str:
    """Check if the model name is valid. Note that this is NOT the huggingface model name, but the model name
    in the model registry (i.e. used by Triton).
    """
    if "/" in model_name or not model_name.islower():
        raise ValueError(
            f"Invalid model name: {model_name}. Must NOT have uppercase letters and must NOT have a '/' in it."
        )
    return model_name


def generate_config_pbtxt(input: DefaultConfigInput, pth_config: Path = None) -> str:
    """Generates a default config.pbtxt file for Triton Inference Server"""
    config = DEFAULT_CONFIG_TEMPLATE.format(**input.dict())
    if pth_config:
        with pth_config.open("w") as f:
            f.write(config)
    return config


def KIV_generate_proto_dir(src_dir: Path = Path("./src").absolute(), force_download: bool = False):
    """[ss] 24/12/2024. Attempt to download the model_config.proto file from the triton-inference-server/common repo and compile it to python using protoc,
    so that we can programmatically create the config.pbtxt file for Triton Inference Server. This is a work in progress and not yet integrated into the main script.
    """
    triton_proto_dir = src_dir / "triton_proto"
    pth_model_config_proto = triton_proto_dir / "raw/model_config.proto"
    triton_proto_dir.mkdir(parents=True, exist_ok=True)
    pth_model_config_proto.parent.mkdir(parents=True, exist_ok=True)

    try:
        from subprocess import check_output

        check_output(["protoc", "--version"])
    except FileNotFoundError:
        raise FileNotFoundError(
            "Protocol Buffers compiler not found. Please install it e.g. `apt install -y protobuf-compiler`"
        )

    if not force_download and pth_model_config_proto.exists():
        print(f"model_config.proto already exists at: {pth_model_config_proto} ; skipping download")
    else:
        import httpx

        resp = httpx.get(
            "https://raw.githubusercontent.com/triton-inference-server/common/main/protobuf/model_config.proto"
        )
        resp.raise_for_status()
        pth_model_config_proto.open("wb").write(resp.content)

    out = check_output(
        [
            "protoc",
            f"--proto_path={pth_model_config_proto.parent.as_posix()}",
            f"--python_out={triton_proto_dir.as_posix()}",
            pth_model_config_proto.as_posix(),
        ]
    )
    assert out == b"", f"Error: {out}"


def KIV_build_pbtxt(models_dir=Path("./models").absolute()):
    """[ss] 24/12/2024. Attempt to programmatically create the config.pbtxt file for Triton Inference Server. This is a work in progress and not yet integrated into the main script."""
    from src.triton_proto import model_config_pb2

    MODEL_NAME = "transformer_model"
    MODEL_VERSION = 1
    MODEL_DIR = models_dir / f"{MODEL_NAME}"
    MODEL_VERSION_DIR = MODEL_DIR / f"{MODEL_VERSION}"
    PTH_MODEL_ONNX = MODEL_DIR / f"{MODEL_VERSION}/model.onnx"
    CONFIG_PATH = MODEL_DIR / "config.pbtxt"

    PTH_MODEL_ONNX.parent.mkdir(parents=True, exist_ok=True)

    # Create model config object
    config = model_config_pb2.ModelConfig()
    config.name = MODEL_NAME
    config.platform = "onnxruntime_onnx"
    config.max_batch_size = 8

    # Input Configurations
    input_ids = config.input.add()
    input_ids.name = "input_ids"
    input_ids.data_type = model_config_pb2.TYPE_INT32
    input_ids.dims.extend([-1])  # Dynamic dimension

    attention_mask = config.input.add()
    attention_mask.name = "attention_mask"
    attention_mask.data_type = model_config_pb2.TYPE_INT32
    attention_mask.dims.extend([-1])  # Dynamic dimension

    # Output Configurations
    output_logits = config.output.add()
    output_logits.name = "logits"
    output_logits.data_type = model_config_pb2.TYPE_FP32
    output_logits.dims.extend([-1])  # Dynamic dimension

    # Instance Group Configuration
    instance_group = config.instance_group.add()
    instance_group.kind = model_config_pb2.ModelInstanceGroup.KIND_GPU
    instance_group.count = 1

    # Dynamic Batching Configuration
    dynamic_batching = model_config_pb2.ModelDynamicBatching()
    dynamic_batching.preferred_batch_size.extend([8, 16])
    dynamic_batching.max_queue_delay_microseconds = 100

    config.dynamic_batching.CopyFrom(dynamic_batching)

    # Write to config.pbtxt
    with open(CONFIG_PATH, "w") as f:
        f.write(str(config))

    print(f"✅ Triton config.pbtxt written to {CONFIG_PATH}")
