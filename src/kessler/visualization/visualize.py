# Event

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, field_validator

from ..io._event import EventDataset
from .__keys import diagonal_features, non_diagonal_features


class Plot(BaseModel):
    events: EventDataset
    diagonal_features: list = diagonal_features

    non_diagonal_features: list = non_diagonal_features

    @field_validator("events")
    def check_events(cls, v):
        if not isinstance(v, EventDataset):
            raise TypeError("events must be of type EventDataset")
        return v

    def plot_feature(
        self,
        feature_name,
        event_ids=None,
        figsize=None,
        ax=None,
        return_ax=False,
        apply_func=None,
        file_name=None,
        legend=False,
        xlim=(-0.01, 7.01),
        ylims=None,
        *args,
        **kwargs,
    ):
        if apply_func is None:
            apply_func = lambda x: x
        events_df = self.events.to_dataframe()

        if event_ids is None:
            event_ids = events_df["EVENT_ID"].unique()

        if isinstance(event_ids, int):
            event_ids = [event_ids]

        for event_id in event_ids:
            event = events_df[events_df["EVENT_ID"] == event_id]

            if ax is None:
                if figsize is None:
                    figsize = 5, 3
                fig, ax = plt.subplots(figsize=figsize)
            ax.plot(
                event["DAYS_TO_TCA"], apply_func(event[feature_name]), marker=".", *args, **kwargs
            )
            ax.set_xlabel("Time to TCA (days)")
            ax.set_title(feature_name)

            if xlim:
                xmin, xmax = xlim
                ax.set_xlim(xmax, xmin)

            if ylims:
                if feature_name in ylims:
                    ymin, ymax = ylims[feature_name]
                    ax.set_ylim(ymin, ymax)

            if legend:
                ax.legend()

            if file_name:
                plt.savefig(file_name)

            if "label" in kwargs:
                kwargs.pop("label")

            if return_ax:
                return ax

    def _title_rows_cols(self, n_features):
        """
        Returns the number of rows and columns for a grid of subplots.
        """
        # Calculate the number of rows and columns based on the number of features
        cols = int(np.ceil(np.sqrt(n_features)))
        rows = int(np.ceil(n_features / cols))
        return rows, cols

    def plot_features(
        self,
        feature_names,
        event_ids=None,
        figsize=None,
        axs=None,
        return_axs=False,
        file_name=None,
        sharex=True,
        *args,
        **kwargs,
    ):
        if not isinstance(feature_names, list):
            feature_names = [feature_names]
        if axs is None:
            rows, cols = self._title_rows_cols(len(feature_names))
            if figsize is None:
                figsize = (cols * 20 / 7, rows * 12 / 6)
            fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=sharex)

        if not isinstance(axs, np.ndarray):
            axs = np.array(axs)
        for i, ax in enumerate(axs.flat):
            if i < len(feature_names):
                if i != 0 and "legend" in kwargs:
                    kwargs["legend"] = False

                self.plot_feature(
                    event_ids=event_ids, feature_name=feature_names[i], ax=ax, *args, **kwargs
                )
            else:
                ax.axis("off")
        plt.tight_layout()

        if file_name:
            plt.savefig(file_name)

        if return_axs:
            return axs

    def plot_uncertainty(
        self,
        event_ids=None,
        object1_prefix="t_",
        object2_prefix="c_",
        figsize=(20, 12),
        diagonal=False,
        *args,
        **kwargs,
    ):
        features = self.diagonal_features if diagonal else self.non_diagonal_features
        features = list(map(lambda f: object1_prefix + f, features)) + list(
            map(lambda f: object2_prefix + f, features)
        )

        return self.plot_features(
            event_ids=event_ids, feature_names=features, figsize=figsize, *args, **kwargs
        )
