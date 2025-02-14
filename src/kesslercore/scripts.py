"""Scripts for the CLI application."""

# ruff: noqa: E402

# %% WARNINGS

import warnings

# disable annoying mlflow warnings
warnings.filterwarnings(action="ignore", category=UserWarning)


# %% PARSERS


# %% SCRIPTS


def main(argv: list[str] | None = None) -> int:
    """Main script for the application."""
    print("Hello, World!")
