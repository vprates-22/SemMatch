"""
report_server.py

This module defines the ReportServer class and a utility function to run a Flask-based report server
that serves visualization and result routes.

The module integrates modular route blueprints into a Flask application using the ReportFactory,
then provides a simple server wrapper and a convenience function to start the server with
custom configurations.

Classes
-------
ReportServer(app: Flask)
    Wraps a Flask application and provides a method to run the development server.

Functions
---------
run_report(config: dict = {}, host: str = 'localhost', port: int = 5000, debug: bool = True) -> None
    Creates and configures the Flask app with routes, then runs the Flask development server.
"""

from flask import Flask
from .factory import ReportFactory
from .routes.result_routes import ResultRoutes
from .routes.visualization_routes import VisualizationRoutes


class ReportServer:
    """
    A simple wrapper around a Flask application to encapsulate server running functionality.

    Attributes
    ----------
    app : Flask
        The Flask application instance to be run.
    """

    def __init__(self, app: Flask):
        self.app = app

    def run(self, host='localhost', port=5000, debug=True) -> None:
        """
        Runs the Flask development server.

        Parameters
        ----------
        host : str, optional
            Hostname to listen on (default is 'localhost').
        port : int, optional
            Port number to listen on (default is 5000).
        debug : bool, optional
            Enables debug mode with hot reloading (default is True).

        Returns
        -------
        None
        """
        self.app.run(host=host, port=port, debug=debug)


def run_report(config: dict = None, host='localhost', port=5000, debug=True) -> None:
    """
    Factory function to create, configure, and run the report Flask server.

    This function initializes the Flask application with the specified route blueprints
    (ResultRoutes and VisualizationRoutes) using the given configuration dictionary,
    then creates a ReportServer instance to run the server.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary passed to route constructors (default is empty dict).
    host : str, optional
        Hostname to listen on (default is 'localhost').
    port : int, optional
        Port number to listen on (default is 5000).
    debug : bool, optional
        Enables debug mode with hot reloading (default is True).

    Returns
    -------
    None
    """
    factory = ReportFactory()
    factory.config_routes(
        # ResultRoutes(config or {}),
        VisualizationRoutes(config or {}),
    )

    server = ReportServer(factory.get_app())
    server.run(host, port, debug)
