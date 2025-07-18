from pathlib import Path
from typing import Optional, Any, List, Union
import numpy as np
import argparse

_type_priorities = [  # In decreasing order
    Path,
    str,
    int,
    float,
    bool,
]


def _priority(o: Any) -> int:
    """Get the priority of an object based on its type."""
    p = next((i for i, t in enumerate(_type_priorities) if type(o) is t), None)
    if p is not None:
        return p
    p = next((i for i, t in enumerate(_type_priorities) if isinstance(o, t)), None)
    if p is not None:
        return p
    return len(_type_priorities)


def print_args(args: argparse.Namespace, parser: Optional[argparse.ArgumentParser] = None) -> None:
    """Print command line arguments in a formatted manner."""
    args_dict = vars(args)
    if parser is None:
        priorities = list(map(_priority, args_dict.values()))
    else:
        all_params = [a.dest for g in parser._action_groups for a in g._group_actions]
        priority = lambda p: all_params.index(p) if p in all_params else len(all_params)
        priorities = list(map(priority, args_dict.keys()))

    pad = max(map(len, args_dict.keys())) + 3
    indices = np.lexsort((list(args_dict.keys()), priorities))
    items = list(args_dict.items())

    print("Arguments:")
    for i in indices:
        param, value = items[i]
        print("    {0}:{1}{2}".format(param, " " * (pad - len(param)), value))
    print("")
