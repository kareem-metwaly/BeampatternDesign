"""contains the project directory class implementation that is helpful to
retrieve many files of the project."""

import typing as t

from utils.base_classes import ConfigType
from utils.etc import get_parent_file, list_files_no_ext, list_files_with_ext, path_to_str
from utils.types import DirectoryPath, FilePath


class ProjectDirectory:
    """Contains all information for all paths of all directories/files in the
    project.

    Helpful when we want to explore a certain place or get absolute
    paths of any location. Should be updated if the structure of the
    project files is changed
    """

    _parent_dir: DirectoryPath = get_parent_file()

    @classmethod
    def configs_path(
        cls, config_type: ConfigType = ConfigType.Undefined, config_name: t.Optional[str] = None
    ) -> t.Union[DirectoryPath, FilePath]:
        """returns the configuration path."""
        path = cls._parent_dir.joinpath("configs")
        if config_type != ConfigType.Undefined:
            path = path.joinpath(config_type.value)
            if config_name:
                path = path.joinpath(config_name + ".yaml")
        else:
            assert not config_name, "You cannot set config_name if you're using undefined config type"

        return path

    @classmethod
    def losses_path(cls) -> DirectoryPath:
        """returns the losses path."""
        path = cls._parent_dir.joinpath("losses")
        return path

    @classmethod
    def models_path(cls) -> DirectoryPath:
        """returns the models path."""
        path = cls._parent_dir.joinpath("models")
        return path

    @classmethod
    def utils_path(cls) -> DirectoryPath:
        """returns the utils path."""
        path = cls._parent_dir.joinpath("utils")
        return path

    @classmethod
    def list_configs(cls, configs_type: ConfigType, no_ext: bool = True, full_path: bool = False) -> list[FilePath]:
        """return iterable of the files of the given config type."""
        if no_ext:
            return list(path_to_str(list_files_no_ext(cls.configs_path(configs_type), ext="yaml"), full_path=full_path))
        return list(path_to_str(list_files_with_ext(cls.configs_path(configs_type), ext="yaml"), full_path=full_path))

    @classmethod
    def logs_path(cls, logger_type: t.Optional[t.Literal["TensorBoard", "MemoryLeak"]]) -> DirectoryPath:
        """where training logs are saved."""
        path = cls._parent_dir.joinpath("logs")
        if logger_type:
            path = path.joinpath(logger_type)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def test_logs_path(cls) -> DirectoryPath:
        """where test logs are saved."""
        path = cls._parent_dir.joinpath("test_logs")
        path.mkdir(parents=True, exist_ok=True)
        return path
