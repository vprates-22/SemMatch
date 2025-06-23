from flask import Flask
from .factory import ReportFactory
from .routes.result_routes import ResultRoutes
from .routes.visualization_routes import VisualizationRoutes

class ReportServer:
    def __init__(self, app:Flask):
        self.app = app

    def run(self, host='localhost', port=5000, debug=True) -> None:
        self.app.run(host=host, port=port, debug=debug)

def runReport(config:dict = {}, host='localhost', port=5000, debug=True) -> None:
    factory = ReportFactory()
    factory.config_routes(
        ResultRoutes(config),
        VisualizationRoutes(config),
    )

    server = ReportServer(factory.get_app())
    server.run(host, port, debug)