import numpy as np
import torch


def total_size(o, handlers={}, verbose=False):
    from sys import getsizeof, stderr
    from itertools import chain
    from collections import deque
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        if isinstance(o, np.ndarray):
            s = o.nbytes
        if isinstance(o, torch.Tensor):
            s = o.storage().size()
        else:
            s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def get_biggest_vars(locals_dict):
    import sys
    var = locals_dict.keys()
    sizes = map(total_size, locals_dict.values())
    sizes_tup = sorted(zip(sizes, var), reverse=True)
    return [(f"{size/10e6:.2f}MB", v) for size, v in sizes_tup][:20]
