from ._base import Service

import typing as T
import pync
import warnings


class AlertsService(Service):
    """Service for sending notifications.

    Require libnotify-bin on Linux systems.

    In production, use with Slack, Discord, or emails.

    https://plyer.readthedocs.io/en/latest/api.html#plyer.facades.Notification

    Parameters:
        enable (bool): use notifications or print.
        app_name (str): name of the application.
        timeout (int | None): timeout in secs.
    """

    enable: bool = True
    app_name: str = "kessler"
    timeout: int | None = None

    @T.override
    def start(self) -> None:
        pass

    def notify(self, title: str, message: str) -> None:
        """Send a notification to the system.

        Args:
            title (str): title of the notification.
            message (str): message of the notification.
        """
        if self.enable:
            try:
                pync.notify(
                    title=title,
                    message=message,
                    app_name=self.app_name,
                    timeout=self.timeout,
                )
            except NotImplementedError:
                warnings.warn(
                    "Notifications are not supported on this system.", RuntimeWarning
                )
                self._print(title=title, message=message)
        else:
            self._print(title=title, message=message)

    def _print(self, title: str, message: str) -> None:
        """Print a notification to the system.

        Args:
            title (str): title of the notification.
            message (str): message of the notification.
        """
        print(f"[{self.app_name}] {title}: {message}")
