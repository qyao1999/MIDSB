import os.path

import os
from collections import OrderedDict
import yaml
from tabulate import tabulate

represent_dict_order = lambda self, data: self.represent_mapping('tag:yaml.org,2002:map', data.items())
yaml.add_representer(OrderedDict, represent_dict_order)

class BaseConfiguer:
    def __init__(self):
        pass

    @classmethod
    def load(cls, file_path: str):
        if '.yaml' not in file_path and '.yml' not in file_path:
            raise ValueError(f'The value of `file_path` should be a path to a yaml file.')
        root_path = os.path.dirname(os.path.abspath(file_path))

        def get_yaml_data(path: str):
            with open(path, encoding='utf-8') as file:
                return yaml.safe_load(file.read())

        def overwrite(ori: dict, con: dict) -> dict:
            _con = OrderedDict()
            _con.update(con)
            for k, v in ori.items():
                if isinstance(v, dict) and k in _con:
                    _con[k] = overwrite(ori[k], _con[k])
                else:
                    _con[k] = v
            return _con

        _origin = get_yaml_data(file_path)
        _config = OrderedDict()
        if 'inherit' in _origin:
            if isinstance(_origin['inherit'], str) or isinstance(_origin['inherit'], list):
                inherit_paths = _origin['inherit']
                if isinstance(inherit_paths, str):
                    inherit_paths = [inherit_paths]

                for inherit_path in inherit_paths:
                    if not os.path.isabs(inherit_path):
                        inherit_path = os.path.join(root_path, inherit_path)
                    _config.update(cls.load(inherit_path))
            else:
                raise TypeError(f'The field of `inherit` in "{os.path.abspath(file_path)}` should be a string or dictionary.')
        for key, value in _origin.items():
            if key not in 'inherit':
                if isinstance(value, dict) and key in _config:
                    _config[key] = overwrite(_origin[key], _config[key])
                else:
                    _config[key] = value
        return _config

    @classmethod
    def dump(cls, data: any, output_path: str):
        with open(output_path, "w", encoding='utf-8') as fo:
            yaml.dump(data, fo, default_flow_style=False)


def read_yml(yml_path):
    yml = BaseConfiguer.load(yml_path)
    return yml


def read_config_from_yaml(config_path: str):
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, 'config.yml')
    if not config_path.endswith('yml') and not config_path.endswith('yaml'):
        raise ValueError(f'The value of `config_path` should be a path to a yaml file, not \'{config_path}\'.')
    if not os.path.exists(config_path):
        raise ValueError(f'The config file `{config_path}` does not exist.')

    config = read_yml(config_path)
    return Config(config)


class Config:
    _MAX_LENGTH = 50
    def __init__(self, config):
        assert isinstance(config, dict), "Config must be a dictionary."
        self.__dict__.update(config)

    def dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def get(self, field, default=None):
        return getattr(self, field, default)

    def update(self, config):
        self.__dict__.update(config)

    def save(self, save_path=None, file_name='config.yml'):
        if save_path is None:
            save_path = self.run_path
        BaseConfiguer.dump(data=self.dict(), output_path=os.path.join(save_path, file_name))

    def handleOvergLength(self, sentence: str, max_length:int) -> dict:
        sentence = sentence if len(sentence) < max_length else sentence[:max_length- 1 - 3] + '...'
        return sentence

    def print(self):
        con = self.dict()
        table_data = []
        for key, value in con.items():
            table_data.append([str(key),self.handleOvergLength(str(value), self._MAX_LENGTH)])
        print('Configuration:')
        print(tabulate(table_data, headers=["Param", "Value"], tablefmt="pretty"))
