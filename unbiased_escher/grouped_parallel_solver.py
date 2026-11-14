"""Use the same structured UCV learner in driver and traversal actors."""
from .hulhe_structured_solver import StructuredHULHEGroupedWideUCVEscher
from .parallel_solver import ParallelUnbiasedControlVariateEscher, UCVEscherTraversalWorker


class StructuredGroupedTraversalWorker(UCVEscherTraversalWorker):
    SOLVER_CLASS = StructuredHULHEGroupedWideUCVEscher


class ParallelStructuredGroupedUCVEscher(
    ParallelUnbiasedControlVariateEscher, StructuredHULHEGroupedWideUCVEscher
):
    def _traversal_worker_class(self):
        return StructuredGroupedTraversalWorker

    def _after_collection_chunk(self):
        self._maybe_run_early_node_checkpoint()

    def iteration(self):
        super().iteration()
        self._maybe_run_training_time_checkpoint()
