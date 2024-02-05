from typing import Any

VALID_KEYS = ['node', 'nodes', 'detector', 'detectors', 'device', 'devices']

def fill_value_by_name(
    config: dict,
    name: str,
    fill_key: str,
    fill_value: Any,
    valid_keys: list[str] = VALID_KEYS,
) -> dict:
  if fill_value is not None:
    for key in valid_keys:
      if key in config:
        for node_config in config[key]:
          if node_config['name'] == name:
            node_config[fill_key] = fill_value
  return config