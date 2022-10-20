"""Contains a custom Yaml loader that has new keywords that are helpful in our
case."""

import typing as t
from pathlib import Path

from utils.types import FilePath
from yaml import SafeLoader as _SafeLoader
from yaml import ScalarNode as _ScalerNode
from yaml import load as _load

_CONSTRUCTOR = t.Callable[["YamlLoader", _ScalerNode], t.Any]
_T = t.TypeVar("_T")

# used to register any new method to be bound with a keyword
_constructors_registry = {}


def _register_constructor(fn_or_name: t.Union[_CONSTRUCTOR, str, None] = None):
    """decorator to register a new yaml constructor.

    :param fn_or_name: either pass the keyword in the yaml file as a str, or it will retrieve it from the name of the
                        method you're decorating it with.
    """

    class Wrapper:
        """used to allow the input to be either a given defined keyword or not
        necessarily having it and automatically retrieve it from the method's
        name."""

        def __init__(self, name: str = None):
            """register the name if given.

            :param name: (str; default: None) if None it will use the name of the method
            """
            self._name = name

        def __call__(self, fn: _CONSTRUCTOR) -> _CONSTRUCTOR:
            """actual registration of the method into the dictionary."""
            assert self._name
            _constructors_registry.update({self._name: fn})
            return fn

    if isinstance(fn_or_name, str | None):
        return Wrapper(fn_or_name)
    else:
        return Wrapper(fn_or_name.__name__)(fn_or_name)


def _bind_constructors(cls: t.Type["YamlLoader"]):
    """a decorator to bind all the registered methods after the definition of
    the class.

    :param cls: (YamlLoader)
    :return: the original input class of type YamlLoader
    """
    for key, method in _constructors_registry.items():
        cls.add_constructor("!" + key, method)
    return cls


@_bind_constructors
class YamlLoader(_SafeLoader):
    """a PyYaml loader that supports some custom keywords. Keywords to be used
    in the yaml file:

    - `!include` to include children files in the parent loaded file.
    - `!fetch` to include the value from the parameters file in the dataset; takes two comma-separated values
               representing the file path (relative to the root directory of the config file) and an expression
               that will be evaluated; expression like 5 * <<K>> will fetch K from the file and multiply by 5.
    """

    _parent: Path
    _root: Path
    _stored: dict[str, t.Any]

    def __init__(self, stream):
        """saves the parent and the root (of all configs) directory. It assumes
        that the configs have two levels structure; meaning we have for
        example: config (parent directory), config_subdir (a subdirectory), and
        a config.yaml (a child of config_subdir).

        :param stream:
        """
        self._parent = Path(stream.name).parent.absolute()
        self._root = self._parent.parent.absolute()
        self._stored = {}
        super().__init__(stream)

    def get_value(self, key):
        """retrieves a value stored in either self._stored or
        self.constructed_objects.

        :param key: the key to look for the value
        :return: the value stored with key
        """
        if key in self._stored:
            return self._stored[key]
        else:
            iterator = iter(self.constructed_objects.values())
            try:
                while next(iterator) != key:
                    ...
                return next(iterator)
            except StopIteration:
                raise ValueError(f"this key is not found in the fetched data so far, key: {key}")

    @classmethod
    def read(cls: t.Type["YamlLoader"], path: FilePath) -> dict[str, t.Any]:
        """
        Read yaml data from path and converts it to dict,
        The file may contain a `!include` keyword to include subfiles
        example:
            `foo.yaml`
            >> a: 1
            >> b:
            >>    - 1.43
            >>    - 543.55
            >> c: !include bar.yaml
            `bar.yaml`
            >> - 3.6
            >> - [1, 2, 3]

        :rtype: dict[str, t.Any]
        :param path: yaml file path e.g. /home/user/data/config.yaml
        :return: dictionary of configs
        """
        with open(path, "r") as stream:
            parsed_yaml = _load(stream, Loader=cls)

        # some values may need to be unrolled as they are constructed by for_each
        parsed_yaml, _ = YamlLoader.unroll(parsed_yaml)
        return parsed_yaml

    @staticmethod
    def unroll(value: _T) -> tuple[_T, bool]:
        """loops recursively and finds !for_each keyword and unroll it.

        :param value: could by anytype
        :return: same type as value
        """
        if isinstance(value, dict):
            if "!for_each" in value.keys():
                assert len(value.keys()) == 1
                return value["!for_each"], True
            else:
                for key in value.keys():
                    value[key], _ = YamlLoader.unroll(value[key])
        elif isinstance(value, list):
            new_value = []
            for idx in range(len(value)):
                v, status = YamlLoader.unroll(value[idx])
                if status:
                    new_value.extend(v)
                else:
                    new_value.append(v)
            value = new_value
        elif isinstance(value, t.Iterable) and not isinstance(value, str):
            raise NotImplementedError(f"Unsupported type for recursive unrolling, {type(value)}")
        return value, False

    @_register_constructor
    def include(self, node: _ScalerNode) -> t.Dict[str, t.Any]:
        """the constructor used with !include keyword.

        :param node: (ScalerNode)
        :return:
        """
        filename = self._parent / self.construct_scalar(node)
        with open(filename, "r") as f:
            return _load(f, YamlLoader)

    @_register_constructor
    def fetch(self, node: _ScalerNode) -> t.Any:
        """the constructor used with !fetch keyword.

        :param node: (ScalerNode) it has to have the format filename.yaml , <<keyword>>
        :return:
        """
        file_name, node = self.construct_scalar(node).split(",")
        file_name = self._root / file_name.strip()
        with open(file_name, "r") as f:
            vals = _load(f, YamlLoader)  # NOQA: F841, vals is actually used later by eval method
        node = node.replace("<<", "vals['").replace(">>", "']")
        return eval(node)

    @_register_constructor
    def store(self, node: _ScalerNode) -> None:
        """stores a fetched value to be loaded later.

        :param node:  it has to have the format filename.yaml , <<keyword>> , given_name
        """
        file_name, node, var_name = self.construct_scalar(node).split(",")
        file_name = self._root / file_name.strip()
        with open(file_name, "r") as f:
            vals = _load(f, YamlLoader)  # NOQA: F841, vals is actually used later by eval method
        node = node.replace("<<", "vals['").replace(">>", "']")
        self._stored[var_name.strip()] = eval(node)

    @_register_constructor
    def load(self, node: _ScalerNode) -> t.Any:
        """loads a pre-stored value. The value could be stored using keyword
        `!store` that reads from another file, or through an already defined
        value in the same file.

        :param node: (ScalerNode) it has to have the format <<var_name>>
        """
        node = self.construct_scalar(node).replace("<<", "self.get_value('").replace(">>", "')")
        return eval(node)

    @_register_constructor
    def store_yaml(self, node: _ScalerNode) -> None:
        """parse a yaml and stores it to be loaded later.

        :param node:  it has to have the format
        !store_yaml
         - key: value (dictionary)
        """
        self._stored.update({self.construct_scalar(key): self.construct_mapping(val) for key, val in node.value})

    @_register_constructor
    def for_each(self, node: _ScalerNode) -> t.Any:
        """runs a yaml for n times.

        :param node: (ScalerNode) it has to have the format !for number_iterations, yaml_stored_variable
        """
        n, yaml_name = self.construct_scalar(node).split(",")
        n = int(n)
        vals = self.get_value(yaml_name.strip())
        out = [vals for _ in range(n)]
        return {"!for_each": out}
