from __future__ import annotations

from typing import Any, Dict, Optional, Union, List
import pickle

import numpy as np
import h5py


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, bool, np.number))


def _write_item(group: h5py.Group, key: str, value: Any, allow_pickle: bool) -> None:
    if not isinstance(key, str):
        raise TypeError(f"Dictionary keys must be str. Invalid key type: {type(key)}")

    if isinstance(value, dict):
        subgroup = group.create_group(key)
        subgroup.attrs["_py_type"] = "dict"
        for subkey, subval in value.items():
            _write_item(subgroup, subkey, subval, allow_pickle)
        return

    if value is None:
        dataset = group.create_dataset(key, shape=(), dtype=np.uint8)
        dataset.attrs["_py_type"] = "none"
        return

    if isinstance(value, str):
        str_dtype = h5py.string_dtype(encoding="utf-8")
        dataset = group.create_dataset(key, data=np.array(value, dtype=str_dtype), dtype=str_dtype)
        dataset.attrs["_py_type"] = "str"
        return

    if _is_scalar(value):
        dataset = group.create_dataset(key, data=value)
        dataset.attrs["_py_type"] = "scalar"
        return

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise TypeError(f"numpy.ndarray with dtype=object is not supported at key '{key}'.")

        if value.dtype.kind in ("U", "S"):
            str_dtype = h5py.string_dtype(encoding="utf-8")
            data = value.astype(str_dtype, copy=False)
            dataset = group.create_dataset(key, data=data, dtype=str_dtype)
            dataset.attrs["_py_type"] = "ndarray_str"
            return

        dataset = group.create_dataset(key, data=value)
        dataset.attrs["_py_type"] = "ndarray"
        return

    if not allow_pickle:
        raise TypeError(
            f"Unsupported object type at key '{key}': {type(value)}. "
            "Enable allow_pickle=True to store arbitrary Python objects."
        )

    pickled = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    data = np.frombuffer(pickled, dtype=np.uint8)
    dataset = group.create_dataset(key, data=data)
    dataset.attrs["_py_type"] = "pickle"


def _read_item(h5obj: Union[h5py.Group, h5py.Dataset]) -> Any:
    if isinstance(h5obj, h5py.Group):
        out: Dict[str, Any] = {}
        for name in h5obj.keys():
            out[name] = _read_item(h5obj[name])
        return out

    py_type = h5obj.attrs.get("_py_type", None)

    if py_type == "none":
        return None

    if py_type == "str":
        val = h5obj[()]
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return str(val)

    if py_type == "scalar":
        val = h5obj[()]
        if isinstance(val, np.generic):
            return val.item()
        return val

    if py_type == "ndarray":
        return h5obj[()]

    if py_type == "ndarray_str":
        arr = h5obj[()]
        if isinstance(arr, np.ndarray) and arr.dtype.kind == "S":
            return arr.astype("U")
        return arr

    if py_type == "pickle":
        buffer = h5obj[()]
        return pickle.loads(buffer.tobytes())

    return h5obj[()]


def save_hdf5(filename: str, data: Dict[str, Any], allow_pickle: bool = False) -> None:
    if not isinstance(data, dict):
        raise TypeError(f"'data' must be a dict, got: {type(data)}")

    with h5py.File(filename, "w") as h5file:
        h5file.attrs["_py_type"] = "dict_root"
        for key, value in data.items():
            _write_item(h5file, key, value, allow_pickle)

def load_hdf5(filename: str, path: Optional[str] = None) -> Any:
    """
    Load data from an HDF5 file created by 'save_dict_to_hdf5'.

    Parameters
    ----------
    filename : str
        Input HDF5 file path.
    path : str or None
        If None, load the full root as a dict.
        If it points to a Group, load that subtree as a dict.
        If it points to a Dataset, load and return that single object.

    Returns
    -------
    Any
        dict or a single reconstructed object.
    """
    with h5py.File(filename, "r") as h5file:
        target: Union[h5py.File, h5py.Group, h5py.Dataset] = h5file

        if path is not None:
            if path not in h5file:
                raise KeyError(f"Path not found in file: {path}")
            target = h5file[path]

        return _read_item(target)


def list_hdf5_paths(filename: str) -> List[str]:
    paths: List[str] = []

    def _visitor(name: str, obj: Union[h5py.Group, h5py.Dataset]) -> None:
        paths.append(f"/{name}")

    with h5py.File(filename, "r") as h5file:
        h5file.visititems(_visitor)

    return sorted(paths)


# class Dummy:
#     def __init__(self, x):
#         self.x = x
#
# data = {
#     "array": np.arange(4),
#     "config": {"alpha": 0.1},
#     "object": Dummy(5),
# }
#
# save_hdf5(data, "test.h5", allow_pickle=True)
#
# print(list_hdf5_paths("test.h5"))
# # ['/array', '/config', '/config/alpha', '/object']
# d   = load_hdf5("test.h5")
# obj = load_hdf5("test.h5", path="/object")      # devuelve Dummy(...)
# cfg = load_hdf5("test.h5", path="/config")  # devuelve {"alpha": 0.1}
