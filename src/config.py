"""Common config parameters for notbooks and scripts"""

from pathlib import Path

models_dir = Path("./models").absolute()


class SimpleNetCfg:
    model_name = "simple_net"
    model_version = "1"
    pth_model = models_dir / f"{model_name}/{model_version}/model.pt"
