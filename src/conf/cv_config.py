from dataclasses import dataclass
import json
from pathlib import Path

from src.definitions import CONFIG_PATH


@dataclass
class RoboflowConfig:
    apiKey: str

    @classmethod
    def from_dict(cls, dict: dict[str, object]):
        return RoboflowConfig(apiKey=dict["apiKey"])


@dataclass
class CvConfig:
    roboflow: RoboflowConfig

    @classmethod
    def from_dict(cls, dict: dict[str, object]):
        return CvConfig(roboflow=RoboflowConfig.from_dict(dict["roboflow"]))


def load_config(path: Path = CONFIG_PATH) -> CvConfig:
    with open(path) as f:
        config = json.load(f)
        return CvConfig.from_dict(config)
