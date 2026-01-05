"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""

import importlib
import pathlib as pl
import sys


class Config(dict):
    """Improved from easydict.whl.

    Support nested dict and list initialization.
    Support nested property access.
    Support nested unpacking assignment.

    Example
    ---
    ```
    cfg = Config(dict(a=1, b=dict(c=2, d=[3, dict(e=4)])))
    print(cfg)  # {'a': 1, 'b': {'c': 2, 'd': [3, {'e': 4}]}}
    a, (c, (d, (e,))) = cfg
    print(a, c, d, e)  # 1, 2, 3, 4
    ```
    """

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)
        for k in self.__class__.__dict__.keys():
            flag1 = k.startswith("__") and k.endswith("__")
            flag2 = k in ("fromfile", "update", "pop")
            if any([flag1, flag2]):
                continue
            setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        super(Config, self).__setattr__(name, value)
        super(Config, self).__setitem__(name, value)

    __setitem__ = __setattr__

    @staticmethod
    def fromfile(cfg_file: pl.Path) -> "Config":
        if isinstance(cfg_file, str):
            cfg_file = pl.Path(cfg_file)
        assert cfg_file.name.endswith(".py")
        assert cfg_file.is_file()
        file_dir = str(cfg_file.absolute().parent)
        fn = str(cfg_file.name).split(".")[0]
        sys.path.append(file_dir)
        module = importlib.import_module(fn)
        # cfg_dict = { k: v for k, v in module.__dict__.items() if not (k.startswith("__") and k.endswith("__")) }
        cfg_dict = module.__dict__
        return Config(cfg_dict)

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(Config, self).pop(k, d)


def build_from_config(cfg):
    """Build a module from config dict."""
    if cfg is None:
        return
    if isinstance(cfg, (list, tuple)):  # iteration
        obj = [build_from_config(_) for _ in cfg]
    elif isinstance(cfg, dict):  # recursion
        cfg = cfg.copy()  # TODO deepcopy ???
        if "type" in cfg:
            cls_key = cfg.pop("type")
        else:
            cls_key = None
        for k, v in cfg.items():
            cfg[k] = build_from_config(v)
        if cls_key is not None:
            obj = cls_key(**cfg)  # MODULE_DICT[cls_key](**cfg)
        else:
            obj = cfg
    else:
        obj = cfg
    return obj


class DictTool:
    """Support nested `dict`s and `list`s."""

    @staticmethod
    def getattr(obj, key):
        assert isinstance(obj, (dict, list))

        def resolve_attr(obj, key):
            keys = key.split(".")
            for name in keys:
                if isinstance(obj, dict):
                    obj = obj.get(name)
                elif isinstance(obj, list) and name.isdigit():
                    obj = obj[int(name)]
                else:
                    raise KeyError(f"Invalid key or index: {name}")
            return obj

        return resolve_attr(obj, key)

    @staticmethod
    def setattr(obj, key, value):
        assert isinstance(obj, (dict, list))

        def resolve_attr(obj, key):
            keys = key.split(".")
            head = keys[:-1]
            tail = keys[-1]
            for name in head:
                if isinstance(obj, dict):
                    if name not in obj:
                        obj[name] = {}
                    obj = obj[name]
                elif isinstance(obj, list) and name.isdigit():
                    idx = int(name)
                    while len(obj) <= idx:
                        obj.append({})
                    obj = obj[idx]
                else:
                    raise KeyError(f"Invalid key or index: {name}")
            return obj, tail

        resolved_obj, resolved_attr = resolve_attr(obj, key)
        if isinstance(resolved_obj, dict):
            resolved_obj[resolved_attr] = value
        elif isinstance(resolved_obj, list) and resolved_attr.isdigit():
            idx = int(resolved_attr)
            while len(resolved_obj) <= idx:
                resolved_obj.append(None)
            resolved_obj[idx] = value
        else:
            raise KeyError(f"Invalid key or index: {resolved_attr}")


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **kwds):
        for t in self.transforms:
            kwds = t(**kwds)
        return kwds

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

    def __getitem__(self, idx):
        return self.transforms[idx]


class ComposeNoStar(Compose):

    def __call__(self, kwds):
        for t in self.transforms:
            kwds = t(kwds)
        return kwds
