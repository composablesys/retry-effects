from collections import defaultdict
from typing import DefaultDict, List, Callable, Tuple
from functools import partial


class ToRestart(Exception):
    pass


class ToResume(Exception):
    pass


class ReturnFromEffect(Exception):
    def __init__(self, val, *args: object) -> None:
        self.val = val
        super().__init__(*args)


class Ouroboros:

    def __init__(self) -> None:
        self.events: DefaultDict[str, List[Callable]] = defaultdict(list)
        self.restarts: DefaultDict[str, List[Callable]] = defaultdict(list)

    def register(self, event_name: str, handler: Callable, func: Callable):
        self.events[event_name].append(handler)
        self.restarts[event_name].append(func)

    def deregister(self, event_name: str):
        self.events[event_name].pop()
        self.restarts[event_name].pop()

    def restart(self):
        raise ToRestart()

    def resume(self):
        raise ToResume()

    def raise_effect(self, event_name, *args, **kwargs):
        try:
            self.events[event_name][-1](*args, **kwargs)
        except ToRestart:
            raise ReturnFromEffect(val=self.restarts[event_name][-1]())
        except ToResume:
            return

    def handle(self, func: callable = None, handlers: List[Tuple[str, Callable]] = None):
        if func is None:
            return partial(
                self.handle,
                handlers=handlers
            )

        def wrapper(*args, **kwargs):
            for event_name, handler in handlers:
                self.register(event_name, handler, lambda: func(*args, **kwargs))
            try:
                return func(*args, **kwargs)
            except ReturnFromEffect as rfe:
                return rfe.val

        return wrapper
