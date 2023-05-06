"""Additional utilities that don't need its own file."""
import collections
import gc
import importlib
import io
import os
import resource
import typing as t
from pathlib import Path

import h5py
import numpy as np
import PIL.Image
import torch
from matplotlib.figure import Figure
from numpy import typing as npt
from torch import Tensor
from torch.nn import Module
from torchvision.transforms.functional import to_pil_image
from utils.types import DirectoryPath, FilePath


def h5_load(path: FilePath, dataset: t.Optional[str] = None) -> t.Union[npt.ArrayLike, dict[str, npt.ArrayLike]]:
    """loads data in an HDF5 file format.

    :param path: the path of the h5 file where data is stored
    :param dataset: (optional default: None) to fetch only specific data from the file
    :return: returns either a numpy array of the requested dataset,
            or a dictionary of numpy arrays of all dataset in the file
    """
    file = h5py.File(path, "r")
    if dataset:
        assert dataset in file.keys(), f"{dataset} is not in this file"
        return np.array(file[dataset])
    data = {}
    for k in file.keys():
        data[k] = np.array(file[k])
    return data


def import_all_submodules(file: str) -> None:
    """import all submodules in the same path of file.

    :param file: (str) the path to import all modules from neighboring files
    :return: None
    """
    # Iterate all files in the same directory
    directory = os.path.dirname(file)
    for neighbor_file in os.listdir(directory):
        # Exclude __init__.py and other non-python files
        if neighbor_file.endswith(".py") and not neighbor_file.startswith("_"):
            # Remove the .py extension
            module_name = neighbor_file[: -len(".py")]
            # Assume src to be the name of the source root directory
            importlib.import_module(
                os.path.relpath(os.path.join(directory, module_name)).replace("\\", ".").replace("/", ".")
            )


def list_files_no_ext(path: DirectoryPath, ext: t.Optional[str] = None) -> t.Iterator[FilePath]:
    """List all files in a directory. names of the files are listed only (not
    including path or extension)

    :param path: (Union[DirectoryPath, str]) the path to explore
    :param ext: (Optional[str], default: None) to filter files and return only with specific extension ``ext``
    :return: (Iterable[str]) of the files in ``path`` directory
    """
    files = (file.with_suffix("") for file in list_files_with_ext(path=path, ext=ext))
    return files


def list_files_with_ext(path: DirectoryPath, ext: t.Optional[str] = None) -> t.Iterator[FilePath]:
    """List all files in a directory. names of the files are listed only with
    extension (not including path)

    :param path: (Union[DirectoryPath, str]) the path to explore
    :param ext: (Optional[str], default: None) to filter files and return only with specific extension ``ext``
    :return: (Generator[FilePath]) of the files in ``path`` directory
    """
    if isinstance(path, str):
        path = Path(path)
    pattern = "*"
    if ext:
        pattern += "." + ext
    files = iter(path.glob(pattern=pattern))
    return files


def list_files(path: DirectoryPath, is_ext: bool, ext: t.Optional[str] = None) -> t.Iterator[FilePath]:
    """List all files in a directory. names of the files are listed with or w/o
    extension (not including path) based on the value of is_ext.

    :param path: (Union[DirectoryPath, str]) the path to explore
    :param is_ext: (bool) whether to include the extension or not
    :param ext: (Optional[str], default: None) to filter files and return only with specific extension ``ext``
    :return: (Generator[FilePath]) of the files in ``path`` directory
    """
    return list_files_no_ext(path=path, ext=ext) if is_ext else list_files_with_ext(path=path, ext=ext)


def path_to_str(paths: t.Iterable[Path], full_path: bool = True) -> t.Iterable[str]:
    """convert the iterable of Path to iterable of strings."""
    if full_path:
        return (str(path) for path in paths)
    return (str(path.name) for path in paths)


def get_parent_file() -> DirectoryPath:
    """retrieves the main directory of unrolled_PDR."""
    return Path(__file__).parent.parent


def find_latest_checkpoint(path: DirectoryPath) -> t.Optional[FilePath]:
    """returns the latest checkpoint in the provided directory.

    Returns None if no checkpoint exists
    """
    files = path.glob(pattern="*.pth")
    try:
        next(files)
    except StopIteration:
        return None
    # TODO: check if max is really what we want
    return max(files)


def fig2img(fig: Figure) -> PIL.Image:
    """Convert a Matplotlib figure to a PIL Image and return it."""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = PIL.Image.open(buf)
    return img


def tensor2img(tensor: Tensor) -> PIL.Image:
    """convert a PyTorch tensor to a Pillow Image."""
    return to_pil_image(tensor)


def show_tensor(tensor: Tensor) -> None:
    """show a tensor as an image."""
    tensor2img(tensor).show()


def freeze_module(module: Module):
    """loops through all parameters of a module and set requires_grad to
    False."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: Module):
    """loops through all parameters of a module and set requires_grad to
    True."""
    for param in module.parameters():
        param.requires_grad = True


def get_num_of_tensors(objects: t.Optional[t.Sequence[t.Any]] = None):
    """finds the number of tensors (inside objects sequence if provided, or
    else globally through the Garbage Collector)

    :param objects: Sequence[Any]; if None, check all objects through the garbage collector
    :return: the number of tensors
    """
    tensors_num = 0
    sizes = collections.Counter()
    objects = objects if objects else gc.get_objects()
    tensors = []
    for obj in objects:
        try:
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                tensors_num += 1
                sizes[obj.size()] += 1
                if not obj.is_cuda:
                    tensors.append(obj)
        except:  # NOQA E722
            pass
    res = sizes[torch.Size([])]
    return res


def get_cpu_mem():
    """get the total usage of the cpu memory."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
