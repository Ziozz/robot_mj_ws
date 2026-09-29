"""OMPL geometric planning backed by the project collision checker."""

from dataclasses import dataclass
import time

import numpy as np
from ompl import base as ob
from ompl import geometric as og

from robot_mj.collision import CollisionPhase, HppFclCollisionChecker


@dataclass(frozen=True)
class GeometricPlan:
    waypoints: np.ndarray
    raw_waypoint_count: int
    simplified_waypoint_count: int
    planning_time: float
    planner_name: str


class OmplPlanner:
    """Plan one joint group without embedding any robot-specific joint names."""

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        checker: HppFclCollisionChecker,
        *,
        phase: CollisionPhase = CollisionPhase.FREE_SPACE,
        joint_positions: dict[str, float] | None = None,
        planner_range: float = 0.22,
        edge_joint_step: float = 0.04,
    ) -> None:
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape or self.lower.ndim != 1:
            raise ValueError("Joint lower/upper limits must be same-length vectors")
        self.checker = checker
        self.phase = phase
        self.joint_positions = joint_positions
        self.edge_joint_step = float(edge_joint_step)

        self.space = ob.RealVectorStateSpace(len(self.lower))
        bounds = ob.RealVectorBounds(len(self.lower))
        for index, (low, high) in enumerate(zip(self.lower, self.upper)):
            bounds.setLow(index, float(low))
            bounds.setHigh(index, float(high))
        self.space.setBounds(bounds)
        # OMPL's discrete motion validator uses Euclidean state-space length.
        # Keeping its segment length <= the required maximum joint step is
        # conservative because max(|dq_i|) <= ||dq||_2.
        extent = float(self.space.getMaximumExtent())
        self.space.setLongestValidSegmentFraction(self.edge_joint_step / extent)

        self.setup = og.SimpleSetup(self.space)
        # OMPL 2.x nanobind accepts a callable directly. Keeping this callback
        # (instead of a Python-derived C++ checker) avoids a reference cycle.
        self.setup.setStateValidityChecker(self._state_is_valid)
        planner = og.RRTConnect(self.setup.getSpaceInformation())
        planner.setRange(float(planner_range))
        self.setup.setPlanner(planner)

    def _array(self, state) -> np.ndarray:
        return np.fromiter((state[i] for i in range(len(self.lower))), dtype=float)

    def _state_is_valid(self, state) -> bool:
        return not self.checker.check(
            self._array(state), self.phase,
            joint_positions=self.joint_positions, stop_at_first=True
        ).in_collision

    def _state(self, values: np.ndarray):
        state = self.space.allocState()
        for index, value in enumerate(values):
            state[index] = float(value)
        return state

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        *,
        timeout: float = 4.0,
        simplify: bool = True,
    ) -> GeometricPlan:
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        expected = self.lower.shape
        if start.shape != expected or goal.shape != expected:
            raise ValueError(f"Start and goal must have shape {expected}")
        if np.any(start < self.lower) or np.any(start > self.upper):
            raise ValueError("Start violates a joint position limit")
        if np.any(goal < self.lower) or np.any(goal > self.upper):
            raise ValueError("Goal violates a joint position limit")
        start_report = self.checker.check(
            start, self.phase, joint_positions=self.joint_positions,
            stop_at_first=True,
        )
        goal_report = self.checker.check(
            goal, self.phase, joint_positions=self.joint_positions,
            stop_at_first=True,
        )
        if start_report.in_collision:
            raise RuntimeError(f"Planning start is in collision: {start_report.contacts}")
        if goal_report.in_collision:
            raise RuntimeError(f"Planning goal is in collision: {goal_report.contacts}")

        self.setup.clear()
        self.setup.setStartAndGoalStates(self._state(start), self._state(goal))
        started = time.perf_counter()
        status = self.setup.solve(float(timeout))
        if not status or not self.setup.haveExactSolutionPath():
            raise RuntimeError(f"OMPL failed to find an exact path in {timeout:.2f} s")
        solution = self.setup.getSolutionPath()
        raw_count = solution.getStateCount()
        if simplify:
            # Cap simplification latency; an uncapped simplifySolution() can
            # spend several seconds improving an already valid short path.
            self.setup.getPathSimplifier().simplify(solution, 0.35)
        waypoints = np.vstack([self._array(state) for state in solution.getStates()])

        # Treat planner and simplifier output as untrusted. Re-run every edge
        # through the exact project checker at the configured angular spacing.
        for index, (first, second) in enumerate(zip(waypoints[:-1], waypoints[1:])):
            if not self.checker.edge_is_valid(
                first,
                second,
                self.phase,
                joint_positions=self.joint_positions,
                max_joint_step=self.edge_joint_step,
            ):
                raise RuntimeError(f"OMPL output edge {index} failed collision recheck")
        return GeometricPlan(
            waypoints=waypoints,
            raw_waypoint_count=raw_count,
            simplified_waypoint_count=len(waypoints),
            planning_time=time.perf_counter() - started,
            planner_name="RRTConnect",
        )
