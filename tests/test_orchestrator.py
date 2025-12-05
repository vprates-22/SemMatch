import numpy as np
import pytest

from semmatch.statistics.orchestrator import PipelineOrchestrator, AnalysisPlan
from semmatch.statistics.pipeline_data import RawDataInput, AnalysisResult
from semmatch.statistics.analyzers.base import DataAnalyzer
from semmatch.statistics.data_generators.base import DataGenerator, GeneratedData
from semmatch.statistics.metrics.base import BaseMetric


class FakeGenerator(DataGenerator):
    def generate(self, raw_data: RawDataInput) -> list[GeneratedData]:
        class Dummy(GeneratedData):
            pass

        return [Dummy()]


class FakeAnalyzer(DataAnalyzer):
    _data_generator_depedencies = [FakeGenerator]
    _data_result_type = AnalysisResult

    def analyze(self, generated_data: dict[type[DataGenerator], list[GeneratedData]]):
        # Return two results with different keys
        return [AnalysisResult(key="r1"), AnalysisResult(key="r2")]


class FakeMetric(BaseMetric):
    _allowed_result_type = AnalysisResult

    def __init__(self, config=None):
        super().__init__(config)
        self._raw_results = []
        self._result = 0.0

    def update(self, data=None, generated_data=None):
        # store the analysis result object for inspection
        self._raw_results.append(data)

    def compute(self) -> None:
        self._result = len(self._raw_results)


class OtherResult:
    """Unrelated result type used to make a metric incompatible with AnalysisResult."""
    pass


class BadMetric(BaseMetric):
    _allowed_result_type = OtherResult

    def __init__(self, config=None):
        super().__init__(config)

    def update(self, data=None):
        pass

    def compute(self) -> None:
        pass


def make_raw_input():
    return RawDataInput(
        image0_path="a/img0.jpg",
        image1_path="b/img1.jpg",
        image0=np.zeros((2, 2)),
        image1=np.zeros((2, 2)),
        mkpts0=np.zeros((1, 2)),
        mkpts1=np.zeros((1, 2)),
        dataset=None,
        pair_index=0,
    )


def test_initialization_instantiates_generators_and_analyzers():
    plan = [AnalysisPlan(title="T1", analysis=FakeAnalyzer,
                         metrics=[FakeMetric])]
    orch = PipelineOrchestrator(plan, {})

    # data_generators should contain an instance of FakeGenerator
    assert any(isinstance(g, FakeGenerator) for g in orch.data_generators)

    # data_analyzers_objects should contain an instance of FakeAnalyzer
    assert any(isinstance(a, FakeAnalyzer)
               for a in orch.data_analyzers_objects)


def test_execute_creates_metrics_and_updates():
    plan = [AnalysisPlan(title="T1", analysis=FakeAnalyzer,
                         metrics=[FakeMetric])]
    orch = PipelineOrchestrator(plan, {})

    raw = make_raw_input()
    orch.execute(raw)

    # After first execute, orchestrator should have created metric objects keyed by result.key
    metric_map = orch.orchestrator_data["T1"][FakeMetric]

    assert "r1" in metric_map and "r2" in metric_map

    # Each metric object should have received one update
    assert len(metric_map["r1"].get_raw_results()) == 1
    assert len(metric_map["r2"].get_raw_results()) == 1


def test_invalid_plan_raises_for_incompatible_metric():
    plan = [AnalysisPlan(
        title="Tbad", analysis=FakeAnalyzer, metrics=[BadMetric])]
    with pytest.raises(ValueError):
        PipelineOrchestrator(plan, {})


def test_multiple_execute_accumulates_results_and_pairs():
    plan = [AnalysisPlan(title="T1", analysis=FakeAnalyzer, metrics=[FakeMetric])]
    orch = PipelineOrchestrator(plan, {})

    raw = make_raw_input()
    orch.execute(raw)
    orch.execute(raw)

    metric_map = orch.orchestrator_data["T1"][FakeMetric]

    # Each metric object should have received two updates (one per execute)
    assert len(metric_map["r1"].get_raw_results()) == 2
    assert len(metric_map["r2"].get_raw_results()) == 2

    # Pairs list should have two entries
    assert len(orch.pairs) == 2


def test_reset_clears_metric_raw_results():
    plan = [AnalysisPlan(title="T1", analysis=FakeAnalyzer, metrics=[FakeMetric])]
    orch = PipelineOrchestrator(plan, {})

    raw = make_raw_input()
    orch.execute(raw)

    metric_map = orch.orchestrator_data["T1"][FakeMetric]
    assert len(metric_map["r1"].get_raw_results()) == 1

    orch.reset()

    # After reset, raw results should be cleared
    assert len(metric_map["r1"].get_raw_results()) == 0


def test_generated_data_is_passed_to_metric_update():
    # Metric that records the generated_data it receives
    class MetricWithGen(FakeMetric):
        def __init__(self, config=None):
            super().__init__(config)
            self.last_generated = None

        def update(self, data=None, generated_data=None):
            super().update(data=data, generated_data=generated_data)
            self.last_generated = generated_data

    plan = [AnalysisPlan(title="T1", analysis=FakeAnalyzer, metrics=[MetricWithGen])]
    orch = PipelineOrchestrator(plan, {})

    raw = make_raw_input()
    orch.execute(raw)

    metric_map = orch.orchestrator_data["T1"][MetricWithGen]
    m = metric_map["r1"]

    # The generated_data mapping should contain the FakeGenerator class as key
    assert any(isinstance(v, list) for v in m.last_generated.values())
    assert list(m.last_generated.keys())[0].__name__ == "FakeGenerator"


def test_single_dependency_handling():
    # Analyzer that declares a single class (not an iterable) as dependency
    class SingleDepAnalyzer(FakeAnalyzer):
        _data_generator_depedencies = FakeGenerator

    plan = [AnalysisPlan(title="T1", analysis=SingleDepAnalyzer, metrics=[FakeMetric])]
    orch = PipelineOrchestrator(plan, {})

    # data_generators should contain an instance of FakeGenerator
    assert any(isinstance(g, FakeGenerator) for g in orch.data_generators)
