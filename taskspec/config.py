import json

class ConfigNotFoundError(Exception):
    def __init__(self, node_name, name):
        full_name = '{}.{}'.format(node_name, name)


        super().__init__(self, 'Configuration `{}` does not exists'.format(full_name))

class ConfigNode:
    def __init__(self, name, data):
        self._name = name
        self._data = data

    def __getattr__(self, name):
        if name not in self._data:
            raise ConfigNotFoundError(self._name, name)

        ret = self._data[name]

        if isinstance(ret, dict):
            return ConfigNode('{}.{}'.format(self._name, name), ret)

        return ret

    def __getitem__(self, name):
        return self.__getattr__(name)

    def __contains__(self, name):
        return name in self._data

class Config(ConfigNode):
    def __init__(self, path):
        self.path = path

        with open(path) as f:
            data = json.load(f)

        super().__init__('', data)

    def save(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self._data, f, indent=4)
