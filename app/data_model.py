from pydantic import BaseModel


class Input_SimpleNet(BaseModel):
    x: list[float]


class Input_finbert_model(BaseModel):
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    trt_model: bool = False


class Input_Finbert(BaseModel):
    text: str
