import typing as tp

@tp.runtime_checkable
class SupportsSub(tp.Protocol):
    def __sub__(self, other: "SupportsSub") -> "SupportsSub": ...

@tp.runtime_checkable
class SupportBool(tp.Protocol):
    def __bool__(self) -> bool: ...