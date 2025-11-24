import numpy as np

from semmatch.configs.base import Config
from semmatch.statistics.metrics import BaseMetric
from semmatch.statistics.pipeline_data import PoseErrorResult


class PoseError(BaseMetric):
    _allowed_result_type = PoseErrorResult

    def __init__(self, config=None):
        super().__init__(config)

    def update(self, data: list[PoseErrorResult]) -> None:
        if isinstance(data, PoseErrorResult):
            data = [data]

        for pose_data in data:
            threshold = pose_data.threshold

            self._raw_results[threshold].append({
                'rotation_errors': pose_data.rotation_errors,
                'translation_errors': pose_data.translation_errors,
            })

    def compute(self) -> None:
        for thresh_key, entries in self._raw_results.items():
            self._result[thresh_key] = {
                "rotation_errors": np.mean(entries['rotation_errors']),
                "translation_errors": np.mean(entries['translation_errors']),
            }


class AUC(BaseMetric):
    _allowed_result_type = PoseErrorResult

    def __init__(self, config=None):
        default_config = Config({
            'pose_thresholds': [5, 10, 20],
        }).merge_config(config)
        super().__init__(default_config)

    def update(self, data: list[PoseErrorResult]) -> None:
        if isinstance(data, PoseErrorResult):
            data = [data]

        for pose_data in data:
            threshold = pose_data.threshold

            self._raw_results[threshold].append({
                'rotation_errors': pose_data.rotation_errors,
                'translation_errors': pose_data.translation_errors,
            })

    def compute(self) -> None:
        pose_thresholds = self._config['pose_thresholds']
        final_results = {}

        for thresh_key, entries in self._raw_results.items():
            # Collect all rotation and translation errors for this threshold
            rot_errs = [
                err for entry in entries for err in entry['rotation_errors']]
            trans_errs = [
                err for entry in entries for err in entry['translation_errors']]
            rot_errs = np.array(rot_errs, dtype=float)
            trans_errs = np.array(trans_errs, dtype=float)

            # Combined pose metric (max(err_t, err_R))
            combined = np.maximum(rot_errs, trans_errs)

            # AUC separately and combined
            auc_rot = self._pose_auc(rot_errs, pose_thresholds)
            auc_trans = self._pose_auc(trans_errs, pose_thresholds)
            auc_comb = self._pose_auc(combined, pose_thresholds)

            # Convert to % for readability
            auc_rot = {t: a * 100 for t, a in zip(pose_thresholds, auc_rot)}
            auc_trans = {t: a * 100 for t,
                         a in zip(pose_thresholds, auc_trans)}
            auc_comb = {t: a * 100 for t, a in zip(pose_thresholds, auc_comb)}

            final_results[thresh_key] = {
                "rotation_auc": auc_rot,
                "translation_auc": auc_trans,
                "combined_auc": auc_comb,
            }

        # Store result in the metric system
        self._result = final_results

    @staticmethod
    def _pose_auc(errors: np.ndarray, thresholds: list[float]) -> list[float]:
        errors = np.asarray(errors)
        if len(errors) == 0:
            return [0.0] * len(thresholds)

        N = len(errors)

        # Ordenação dos erros
        errors_sorted = np.sort(errors)
        recalls = (np.arange(1, N+1)) / N  # fração de keypoints corretos

        # Base da curva (0,0) até todos os erros
        base_e = np.r_[0.0, errors_sorted]
        base_r = np.r_[0.0, recalls]

        aucs = []
        for t in thresholds:
            if t == 0:
                aucs.append(0.0)
                continue

            # índice onde t deve entrar na curva
            idx = np.searchsorted(base_e, t, side='right')

            # construir a curva até t
            if idx == 0:
                e = np.array([0.0, t])
                r = np.array([0.0, 0.0])
            else:
                e = np.r_[base_e[:idx], t]
                r = np.r_[base_r[:idx], base_r[idx-1]]

            # calcular AUC normalizado
            auc = np.trapz(r, x=e) / t
            aucs.append(float(auc))

        return aucs
