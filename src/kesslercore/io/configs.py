"""Parse, merge, and convert config objects."""

# %% IMPORTS

import typing as T

import omegaconf as oc

# %% TYPES

Config_ = oc.ListConfig | oc.DictConfig

# %% PARSERS


class Config:
    @staticmethod
    def parse_file(path: str) -> Config_:
        """Parse a config file from a path.

        Args:
            path (str): path to local config.

        Returns:
            Config: representation of the config file.
        """
        return oc.OmegaConf.load(path)

    @staticmethod
    def parse_string(string: str) -> Config_:
        """Parse the given config string.

        Args:
            string (str): content of config string.

        Returns:
            Config: representation of the config string.
        """
        return oc.OmegaConf.create(string)

    # %% MERGERS
    @staticmethod
    def merge_configs(configs: T.Sequence[Config_]) -> Config_:
        """Merge a list of config into a single config.

        Args:
            configs (T.Sequence[Config]): list of configs.

        Returns:
            Config: representation of the merged config objects.
        """
        return oc.OmegaConf.merge(*configs)

    # %% CONVERTERS
    @staticmethod
    def to_object(config: Config_, resolve: bool = True) -> object:
        """Convert a config object to a python object.

        Args:
            config (Config): representation of the config.
            resolve (bool): resolve variables. Defaults to True.

        Returns:
            object: conversion of the config to a python object.
        """
        return oc.OmegaConf.to_container(config, resolve=resolve)
