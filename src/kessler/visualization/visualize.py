# Event
def plot_feature(
    self,
    feature_name,
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
    data_x = []
    data_y = []
    for i, cdm in enumerate(self._cdms):
        if cdm["TCA"] is None:
            raise RuntimeError("CDM {} in event does not have TCA".format(i))
        if cdm["CREATION_DATE"] is None:
            raise RuntimeError("CDM {} in event does not have CREATION_DATE".format(i))
        time_to_tca = util.from_date_str_to_days(cdm["TCA"], date0=cdm["CREATION_DATE"])
        data_x.append(time_to_tca)
        data_y.append(apply_func(cdm[feature_name]))
    # Creating axes instance
    if ax is None:
        if figsize is None:
            figsize = 5, 3
        fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data_x, data_y, marker=".", *args, **kwargs)
    # ax.scatter(data_x, data_y)
    ax.set_xlabel("Time to TCA (days)")
    ax.set_title(feature_name)

    # xmin, xmax = min(ax.get_xlim()), max(ax.get_xlim())
    # ax.set_xlim(xmax, xmin)
    if xlim is not None:
        xmin, xmax = xlim
        ax.set_xlim(xmax, xmin)

    if ylims is not None:
        if feature_name in ylims:
            ymin, ymax = ylims[feature_name]
            ax.set_ylim(ymin, ymax)

    if legend:
        ax.legend()

    if file_name is not None:
        print("Plotting to file: {}".format(file_name))
        plt.savefig(file_name)

    if return_ax:
        return ax


def plot_features(
    self,
    feature_names,
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
        rows, cols = util.tile_rows_cols(len(feature_names))
        if figsize is None:
            figsize = (cols * 20 / 7, rows * 12 / 6)
        fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=sharex)

    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)
    for i, ax in enumerate(axs.flat):
        if i < len(feature_names):
            if i != 0 and "legend" in kwargs:
                kwargs["legend"] = False

            self.plot_feature(feature_names[i], ax=ax, *args, **kwargs)
        else:
            ax.axis("off")
    plt.tight_layout()

    if file_name is not None:
        print("Plotting to file: {}".format(file_name))
        plt.savefig(file_name)

    if return_axs:
        return axs


def plot_uncertainty(self, figsize=(20, 12), diagonal=False, *args, **kwargs):
    if diagonal:
        features = ["CR_R", "CT_T", "CN_N", "CRDOT_RDOT", "CTDOT_TDOT", "CNDOT_NDOT"]
    else:
        features = [
            "CR_R",
            "CT_R",
            "CT_T",
            "CN_R",
            "CN_T",
            "CN_N",
            "CRDOT_R",
            "CRDOT_T",
            "CRDOT_N",
            "CRDOT_RDOT",
            "CTDOT_R",
            "CTDOT_T",
            "CTDOT_N",
            "CTDOT_RDOT",
            "CTDOT_TDOT",
            "CNDOT_R",
            "CNDOT_T",
            "CNDOT_N",
            "CNDOT_RDOT",
            "CNDOT_TDOT",
            "CNDOT_NDOT",
        ]
    features = list(map(lambda f: "OBJECT1_" + f, features)) + list(
        map(lambda f: "OBJECT2_" + f, features)
    )
    return self.plot_features(features, figsize=figsize, *args, **kwargs)


#EventDataset


def plot_event_lengths(self, figsize=(6, 4), file_name=None, *args, **kwargs):
        fig, ax = plt.subplots(figsize=figsize)
        event_lengths = self.event_lengths()
        ax.hist(event_lengths, *args, **kwargs)
        ax.set_xlabel("Event length (number of CDMs)")
        if file_name is not None:
            print("Plotting to file: {}".format(file_name))
            plt.savefig(file_name)

    def plot_feature(
        self,
        feature_name,
        figsize=None,
        ax=None,
        return_ax=False,
        file_name=None,
        *args,
        **kwargs,
    ):
        if ax is None:
            if figsize is None:
                figsize = 5, 3
            fig, ax = plt.subplots(figsize=figsize)
        for event in self:
            event.plot_feature(feature_name, ax=ax, *args, **kwargs)
            if "label" in kwargs:
                kwargs.pop(
                    "label"
                )  # We want to label only the first Event in this EventDataset, for not cluttering the legend

        if file_name is not None:
            print("Plotting to file: {}".format(file_name))
            plt.savefig(file_name)

        if return_ax:
            return ax

    def plot_features(
        self,
        feature_names,
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
            rows, cols = util.tile_rows_cols(len(feature_names))
            if figsize is None:
                figsize = (cols * 20 / 7, rows * 12 / 6)
            fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=sharex)

        if not isinstance(axs, np.ndarray):
            axs = np.array(axs)
        for i, ax in enumerate(axs.flat):
            if i < len(feature_names):
                if i != 0 and "legend" in kwargs:
                    kwargs["legend"] = False

                self.plot_feature(feature_names[i], ax=ax, *args, **kwargs)
            else:
                ax.axis("off")
        plt.tight_layout()

        if file_name is not None:
            print("Plotting to file: {}".format(file_name))
            plt.savefig(file_name)

        if return_axs:
            return axs

    def plot_uncertainty(self, figsize=(20, 12), diagonal=False, *args, **kwargs):
        if diagonal:
            features = [
                "CR_R",
                "CT_T",
                "CN_N",
                "CRDOT_RDOT",
                "CTDOT_TDOT",
                "CNDOT_NDOT",
            ]
        else:
            features = [
                "CR_R",
                "CT_R",
                "CT_T",
                "CN_R",
                "CN_T",
                "CN_N",
                "CRDOT_R",
                "CRDOT_T",
                "CRDOT_N",
                "CRDOT_RDOT",
                "CTDOT_R",
                "CTDOT_T",
                "CTDOT_N",
                "CTDOT_RDOT",
                "CTDOT_TDOT",
                "CNDOT_R",
                "CNDOT_T",
                "CNDOT_N",
                "CNDOT_RDOT",
                "CNDOT_TDOT",
                "CNDOT_NDOT",
            ]
        features = list(map(lambda f: "OBJECT1_" + f, features)) + list(
            map(lambda f: "OBJECT2_" + f, features)
        )
        return self.plot_features(features, figsize=figsize, *args, **kwargs)