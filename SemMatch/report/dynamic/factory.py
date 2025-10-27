"""
Module: report.factory

This module provides the ReportFactory class to facilitate the creation and configuration
of a Flask application with modular route blueprints.

The ReportFactory class:
- Initializes a Flask application instance.
- Allows easy registration of multiple Flask Blueprint routes.
- Provides a method to retrieve the configured Flask app.

This design helps in organizing route definitions into separate modules (blueprints)
and then combining them into a single Flask app, promoting modularity and maintainability.
"""
from flask import Flask


class ReportFactory:
    """
    A factory class to create and configure a Flask application by registering route blueprints.

    This class encapsulates the Flask app initialization and provides a method to
    register multiple route blueprints at once. It simplifies managing multiple
    route modules and consolidates them into a single Flask app instance.

    Attributes
    ----------
    app : Flask
        The Flask application instance managed by this factory.
    """

    def __init__(self):
        self.app = Flask(__name__)

    def config_routes(self, *routes) -> None:
        """
        Registers multiple route blueprints with the Flask application.

        Parameters
        ----------
        *routes : variable number of route objects
            Each route object is expected to have a 'blueprint' attribute
            which is a Flask Blueprint instance to be registered.

        Returns
        -------
        None
        """
        for route in routes:
            self.app.register_blueprint(route.blueprint)

    def get_app(self) -> Flask:
        """
        Returns the configured Flask application instance.

        Returns
        -------
        Flask
            The Flask app with all registered blueprints.
        """
        return self.app
