# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import traceback
from typing import Any, Callable, Generic

from gepa.core.state import GEPAState, initialize_gepa_state
from gepa.logging.utils import log_detailed_metrics_after_discovering_new_program
from gepa.proposer.merge import MergeProposer
from gepa.proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer

from .adapter import DataInst, RolloutOutput, Trajectory

# Import tqdm for progress bar functionality
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class GEPAEngine(Generic[DataInst, Trajectory, RolloutOutput]):
    """
    Orchestrates the optimization loop. It uses pluggable ProposeNewCandidate strategies.
    """

    def __init__(
        self,
        run_dir: str | None,
        evaluator: Callable[[list[DataInst], dict[str, str]], tuple[list[RolloutOutput], list[float]]],
        valset: list[DataInst] | None,
        seed_candidate: dict[str, str],
        # Controls
        max_metric_calls: int | None,
        perfect_score: float,
        seed: int,
        # Strategies and helpers
        reflective_proposer: ReflectiveMutationProposer,
        merge_proposer: MergeProposer | None,
        # Logging
        logger: Any,
        experiment_tracker: Any,
        track_best_outputs: bool = False,
        display_progress_bar: bool = False,
        raise_on_exception: bool = True,
    ):
        # Budget constraint: max_metric_calls must be set
        assert max_metric_calls is not None, "max_metric_calls must be set"

        self.logger = logger
        self.run_dir = run_dir
        self.evaluator = evaluator
        self.valset = valset
        self.seed_candidate = seed_candidate

        self.max_metric_calls = max_metric_calls

        self.perfect_score = perfect_score
        self.seed = seed
        self.experiment_tracker = experiment_tracker

        self.reflective_proposer = reflective_proposer
        self.merge_proposer = merge_proposer

        # Merge scheduling flags (mirroring previous behavior)
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

        self.track_best_outputs = track_best_outputs
        self.display_progress_bar = display_progress_bar

        self.raise_on_exception = raise_on_exception

    def _val_evaluator(self) -> Callable[[dict[str, str]], tuple[list[RolloutOutput], list[float]]]:
        assert self.valset is not None
        return lambda prog: self.evaluator(self.valset, prog)

    def _get_pareto_front_programs(self, state: GEPAState) -> list:
        return state.program_at_pareto_front_valset

    def _run_full_eval_and_add(
        self,
        new_program: dict[str, str],
        state: GEPAState,
        parent_program_idx: list[int],
    ) -> tuple[int, int]:
        num_metric_calls_by_discovery = state.total_num_evals

        valset_outputs, valset_subscores = self._val_evaluator()(new_program)
        valset_score = sum(valset_subscores) / len(valset_subscores)

        state.num_full_ds_evals += 1
        state.total_num_evals += len(valset_subscores)

        new_program_idx, linear_pareto_front_program_idx = state.update_state_with_new_program(
            parent_program_idx=parent_program_idx,
            new_program=new_program,
            valset_score=valset_score,
            valset_outputs=valset_outputs,
            valset_subscores=valset_subscores,
            run_dir=self.run_dir,
            num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery,
        )
        state.full_program_trace[-1]["new_program_idx"] = new_program_idx

        if new_program_idx == linear_pareto_front_program_idx:
            self.logger.log(f"Iteration {state.i + 1}: New program is on the linear pareto front")

        log_detailed_metrics_after_discovering_new_program(
            logger=self.logger,
            gepa_state=state,
            valset_score=valset_score,
            new_program_idx=new_program_idx,
            valset_subscores=valset_subscores,
            experiment_tracker=self.experiment_tracker,
            linear_pareto_front_program_idx=linear_pareto_front_program_idx,
        )
        return new_program_idx, linear_pareto_front_program_idx

    def run(self) -> GEPAState:
        # Check tqdm availability if progress bar is enabled
        current_cheatsheet = self.seed_candidate["instruction_prompt"]
        final_cheatsheet = self.reflective_proposer.propose(current_cheatsheet)

        return final_cheatsheet
