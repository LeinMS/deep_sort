import yaml, json, os

class Config:
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v)


def load_config(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path, 'r') as f:
        if ext in ['.yml', '.yaml']:
            cfg = yaml.safe_load(f)
        elif ext == '.json':
            cfg = json.load(f)
        else:
            raise ValueError('Unsupported config format')
    return Config(cfg)