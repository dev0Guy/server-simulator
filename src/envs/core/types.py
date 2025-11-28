import typing as tp


@tp.runtime_checkable
class SupportsSub(tp.Protocol):
    def __sub__(self, other: "SupportsSub") -> "SupportsSub": ...
