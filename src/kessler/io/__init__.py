from .example import ExampleReader, ExampleWriter
from .configs import Config

ReaderKind = ExampleReader
WriterKind = ExampleWriter

__all__ = ["Config", "ExampleReader", "ExampleWriter"]
