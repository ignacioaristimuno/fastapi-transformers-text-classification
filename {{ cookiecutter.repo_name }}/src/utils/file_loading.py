import json
import yaml

from src.settings import custom_logger


logger = custom_logger("File loading utils")


def read_yaml_file(file_path: str, key: str = "") -> dict:
    """
    Function for reading a yaml file and returning its content as a dictionary.
    If a key is provided, it will return
    """

    try:
        with open(file_path, "r") as file:
            configs = yaml.safe_load(file)
        if not key:
            return configs
        elif key not in configs.keys():
            logger.info(f"Key '{key}' not found in file {file_path}")
            return configs
        return configs[key]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File {file_path} not found") from e


def read_json_file(file_path: str) -> dict:
    """Function for reading a json file and returning its content as a dictionary."""

    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File {file_path} not found") from e
