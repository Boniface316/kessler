from .example import (
    CustomLoader,
    CustomSaver,
    MlflowRegister,
    BuiltinLoader,
    BuiltinSaver,
)

SaverKind = CustomSaver | BuiltinSaver
LoaderKind = CustomLoader | BuiltinLoader
RegisterKind = MlflowRegister


__all__ = [
    "CustomSaver",
    "BuiltinSaver",
    "SaverKind",
    "CustomLoader",
    "BuiltinLoader",
    "LoaderKind",
    "MlflowRegister",
    "RegisterKind",
]
