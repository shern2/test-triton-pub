"""

References:
https://blog.marvik.ai/2023/10/16/deploying-llama2-with-nvidia-triton-inference-server/
"""

from pathlib import Path

import numpy as np

# NOTE: this is only available when Triton runs it
import triton_python_backend_utils as pb_utils  # type: ignore
from transformers import AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        """
        Initialize the tokenizer.
        """
        model_dir = Path(args["model_repository"])
        pth_tokenizer = model_dir / str(args["model_version"]) / "tokenizer_data"
        if not pth_tokenizer.exists():
            raise FileNotFoundError(f"Tokenizer files expected at: {pth_tokenizer}")

        self.tokenizer = AutoTokenizer.from_pretrained(pth_tokenizer)

    def execute(self, requests):
        """
        Tokenize input text for all requests in a single batch.
        """
        # Collect all input texts
        # note: Triton preserves the tensor shape even for individual requests
        # i.e. (1,1) when max_batch_size>0 set in config.pbtxt
        input_texts = [
            pb_utils.get_input_tensor_by_name(request, "text").as_numpy()[0][0].decode("utf-8") for request in requests
        ]
        # pb_utils.Logger.log_info(f"Received input texts: {input_texts}")

        tokens = self.tokenizer(
            input_texts,
            return_tensors="np",
            # NOTE: no padding/truncation specified here, following examples:
            # https://github.com/triton-inference-server/tensorrtllm_backend/blob/7a56e091a788ccf042760cf2c63ea957efc398db/all_models/inflight_batcher_llm/preprocessing/1/model.py#L107-L110
        )
        input_ids = tokens["input_ids"].astype(np.int32)
        attention_mask = tokens["attention_mask"].astype(np.int32)
        token_type_ids = tokens["token_type_ids"].astype(np.int32)

        # pb_utils.Logger.log_info(f"Tokenized input_ids: {input_ids.shape}")

        responses = [
            pb_utils.InferenceResponse(
                # note: when max_batch_size>0 set in config.pbtxt, each request must have the shape (1, ...) to align with the protobuf schema
                output_tensors=[
                    pb_utils.Tensor("input_ids", np.expand_dims(input_ids[i], 0)),
                    pb_utils.Tensor("attention_mask", np.expand_dims(attention_mask[i], 0)),
                    pb_utils.Tensor("token_type_ids", np.expand_dims(token_type_ids[i], 0)),
                ]
            )
            for i in range(len(requests))
        ]
        return responses

    def finalize(self):
        """
        Clean up when Triton shuts down.
        """
        pass
