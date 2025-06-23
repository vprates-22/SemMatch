from flask import Flask

class ReportFactory:
    def __init__(self):
        self.app = Flask(__name__)

    def config_routes(self, *routes) -> None:
        for route in routes:
            self.app.register_blueprint(route.blueprint)

    def get_app(self) -> Flask:
        return self.app