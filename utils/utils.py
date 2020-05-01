import re, importlib, os, sys, types
from matplotlib import pyplot as plt
from typing import Any, List, Tuple, Union

def plotLossCurve(opts, Loss_D_list, Loss_G_list):
    plt.figure()
    plt.plot(Loss_D_list, '-')
    plt.title("Loss curve: Discriminator")
    plt.savefig(os.path.join(opts.det, 'images', 'loss_curve_discriminator.png'))

    plt.figure()
    plt.plot(Loss_G_list, '-o')
    plt.title("Loss curve: Generator")
    plt.savefig(os.path.join(opts.det, 'images', 'loss_curve_generator.png'))

def topFunction(obj: Any) -> str:
    return obj.__module__+'.'+obj.__name__

def modulefromObjname(obj_name: str) -> Tuple[types.ModuleType, str]:
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    parts = obj_name.split(".")
    name_pairs = [(".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)]

    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)
            getObjmodule(module, local_obj_name)
            return module, local_obj_name
        except:
            pass

    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name)
        except ImportError:
            if not str(sys.exc_info()[1]).startswith("No module named '" + module_name + "'"):
                raise

    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)
            getObjmodule(module, local_obj_name)
        except ImportError:
            pass

    raise ImportError(obj_name)

def getObjmodule(module: types.ModuleType, obj_name: str) -> Any:
    if obj_name == '':
        return module
    o = module
    for i in obj_name.split("."):
        o = getattr(o, i)
    return o

def getObjname(name: str) -> Any:
    module, obj_name = modulefromObjname(name)
    return getObjmodule(module, obj_name)


def call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    assert func_name is not None
    func_obj = getObjname(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)
