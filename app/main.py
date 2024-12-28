import logging
import uuid

import numpy as np
import tritonclient.grpc.aio as grpcclient
from data_model import Input_Finbert, Input_finbert_model, Input_SimpleNet
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from tritonclient.grpc.aio import InferenceServerException

# gRPC endpoint for Triton Inference Server
TRITON_SERVER_URL = "localhost:8001"

logger = logging.getLogger("predict_logger")

app = FastAPI()


@app.get("/health")
@app.get("/healthz")
async def health():
    try:
        async with grpcclient.InferenceServerClient(url=TRITON_SERVER_URL) as client:
            if not await client.is_server_ready():
                return JSONResponse(content={"status": "NOT OK"}, status_code=503)
        return JSONResponse(content={"status": "OK"}, status_code=200)
    except InferenceServerException as e:
        return JSONResponse(content={"status": "NOT OK", "error": str(e)}, status_code=503)


# TODO [ss] 27/12/2024. Implement the following endpoints:
# curl -X POST http://127.0.0.1:8000/v2/repository/index
# curl -X GET http://127.0.0.1:8000/v2/models/finbert-trt-model/config


@app.post("/models/simple_net")
async def predict__simple_net(data: Input_SimpleNet):
    try:
        async with grpcclient.InferenceServerClient(url=TRITON_SERVER_URL) as client:
            # Prepare input tensor
            triton_input = grpcclient.InferInput("x", [len(data.x)], "FP32")
            triton_input.set_data_from_numpy(np.array(data.x, dtype=np.float32))

            # Perform inference
            response = await client.infer(
                model_name="simple_net",
                inputs=[triton_input],
                outputs=[grpcclient.InferRequestedOutput("output")],
                request_id=str(uuid.uuid4()),
            )

            result = response.as_numpy("output").tolist()
            return JSONResponse(content={"prediction": result}, status_code=200)

    except InferenceServerException as e:
        return JSONResponse(
            content={"error": "Failed to query Triton Inference Server", "details": str(e)}, status_code=500
        )
    except Exception as e:
        logging.error(f"{e.__class__.__name__}: {e}", exc_info=True)
        return JSONResponse(content={"error": "Unexpected error", "details": str(e)}, status_code=500)


@app.post("/models/finbert-model")
async def predict__finbert_model(data: Input_finbert_model):
    """The direct model endpoint for FinBERT (i.e. inputs are already tokenized)"""
    try:
        async with grpcclient.InferenceServerClient(url=TRITON_SERVER_URL) as client:
            # Prepare input tensors
            input_ids = grpcclient.InferInput("input_ids", [1, len(data.input_ids)], "INT32")
            input_ids.set_data_from_numpy(np.array([data.input_ids], dtype=np.int32))

            attention_mask = grpcclient.InferInput("attention_mask", [1, len(data.attention_mask)], "INT32")
            attention_mask.set_data_from_numpy(np.array([data.attention_mask], dtype=np.int32))

            token_type_ids = grpcclient.InferInput("token_type_ids", [1, len(data.token_type_ids)], "INT32")
            token_type_ids.set_data_from_numpy(np.array([data.token_type_ids], dtype=np.int32))

            # Perform inference
            response = await client.infer(
                model_name="finbert-trt-model" if data.trt_model else "finbert-model",
                inputs=[input_ids, attention_mask, token_type_ids],
                outputs=[grpcclient.InferRequestedOutput("logits")],
                request_id=str(uuid.uuid4()),
            )
            return JSONResponse(content={"logits": response.as_numpy("logits").tolist()}, status_code=200)

    except InferenceServerException as e:
        return JSONResponse(
            content={"error": "Failed to query Triton Inference Server", "details": str(e)}, status_code=500
        )
    except Exception as e:
        logging.error(f"{e.__class__.__name__}: {e}", exc_info=True)
        return JSONResponse(content={"error": "Unexpected error", "details": str(e)}, status_code=500)


@app.post("/models/finbert-tokenizer")
async def predict__finbert_tokenizer(data: Input_Finbert):
    try:
        async with grpcclient.InferenceServerClient(url=TRITON_SERVER_URL) as client:
            # Prepare input tensor
            text = np.array([[data.text]], dtype=np.object_)
            triton_input = grpcclient.InferInput("text", text.shape, "BYTES")
            triton_input.set_data_from_numpy(text)

            response = await client.infer(
                model_name="finbert-tokenizer",
                inputs=[triton_input],
                outputs=[
                    grpcclient.InferRequestedOutput("input_ids"),
                    grpcclient.InferRequestedOutput("attention_mask"),
                    grpcclient.InferRequestedOutput("token_type_ids"),
                ],
                request_id=str(uuid.uuid4()),
            )

            return JSONResponse(
                content={
                    "input_ids": response.as_numpy("input_ids").tolist(),
                    "attention_mask": response.as_numpy("attention_mask").tolist(),
                    "token_type_ids": response.as_numpy("token_type_ids").tolist(),
                },
                status_code=200,
            )

    except InferenceServerException as e:
        return JSONResponse(
            content={"error": "Failed to query Triton Inference Server", "details": str(e)}, status_code=500
        )
    except Exception as e:
        logging.error(f"{e.__class__.__name__}: {e}", exc_info=True)
        return JSONResponse(content={"error": "Unexpected error", "details": str(e)}, status_code=500)


@app.post("/models/finbert")
async def predict__finbert(data: Input_Finbert):
    logger.info(f"Received input: {data}")
    try:
        async with grpcclient.InferenceServerClient(url=TRITON_SERVER_URL) as client:
            # Prepare input tensor
            text = np.array([[data.text]], dtype=np.object_)
            triton_input = grpcclient.InferInput("text", text.shape, "BYTES")
            triton_input.set_data_from_numpy(text)

            response = await client.infer(
                model_name="finbert",
                inputs=[triton_input],
                outputs=[grpcclient.InferRequestedOutput("logits")],
                request_id=str(uuid.uuid4()),
            )
            return JSONResponse(content={"logits": response.as_numpy("logits").tolist()}, status_code=200)

    except InferenceServerException as e:
        return JSONResponse(
            content={"error": "Failed to query Triton Inference Server", "details": str(e)}, status_code=500
        )
    except Exception as e:
        logging.error(f"{e.__class__.__name__}: {e}", exc_info=True)
        return JSONResponse(content={"error": "Unexpected error", "details": str(e)}, status_code=500)
