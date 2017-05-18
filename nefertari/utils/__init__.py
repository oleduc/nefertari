import threading
import os
from abc import abstractclassmethod

from nefertari.utils.data import *
from nefertari.utils.dictset import *
from nefertari.utils.utils import *

_split = split_strip


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ThreadLocalSingletonMeta(type):
    _storage = {}

    def __call__(cls, *args, **kwargs):
        process_id = os.getpid()
        thread_id = threading.get_ident()

        if process_id not in cls._storage:
            cls._storage[process_id] = {}

        if thread_id not in cls._storage[process_id]:
            cls._storage[process_id][thread_id] = {}

        if cls not in cls._storage[process_id][thread_id]:
            cls._storage[process_id][thread_id][cls] = super(ThreadLocalSingletonMeta, cls).__call__(*args, **kwargs)
        return cls._storage[process_id][thread_id][cls]
