from typing import Any, List, Optional, Sequence, Tuple
import math
import numpy as np
import quaternion as qt
from habitat.config import Config, read_write
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
)
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    Simulator,
)
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.nav import (
    Success,
    DistanceToGoal,
    NavigationEpisode
)
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    angle_between_quaternions,
)
from src.sensor import ImageGoalSensorV2
cv2 = try_cv2_import()


@registry.register_measure
class OrienToGoal(Measure):
    """The measure calculates a orientation towards the goal."""

    cls_uuid: str = "orien_to_goal"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position: qt.quaternion = None
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_rotation = None
        self._metric = None
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ):
        current_rotation = self._sim.get_agent_state().rotation

        if self._previous_rotation is None or not np.allclose(
            self._previous_rotation, current_rotation, atol=1e-4
        ):
            r = episode.goals[0].rotation
            gr = qt.quaternion(*(r[-1:]+r[:-1]))
            # rotation: [x, y, z, w] = [0, np.sin(euler / 2), 0, np.cos(euler / 2)]
            cr = current_rotation
            dr = qt.as_euler_angles(cr)[1] - qt.as_euler_angles(gr)[1]
            orien_to_target = math.fabs(
                math.pi / 2 - dr if dr > math.pi / 2 else dr
            ) # * 180 / math.pi
            self._previous_rotation = current_rotation
            self._metric = orien_to_target


@registry.register_measure
class ZERReward(Measure):

    cls_uuid: str = "zer_reward"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._previous_distance: Optional[float] = None
        self._gamma = 0.01
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._previous_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        # FIXME use a metric to calculate orien
        self._previous_orien = task.measurements.measures[
            OrienToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        reduced_distance_to_target = self._previous_distance - distance_to_target
        self._previous_distance = distance_to_target

        self._metric = reduced_distance_to_target

        if distance_to_target < task._config.success.success_distance:
            orien_to_target = task.measurements.measures[
                OrienToGoal.cls_uuid
            ].get_metric()
            reduced_orien_to_target = self._previous_orien - orien_to_target
            self._previous_orien = orien_to_target

            self._metric += reduced_orien_to_target
        
        self._metric -= self._gamma


@registry.register_measure
class ViewAngle(Measure):
    r"""The angle between the agent pose and the goal view when stopping
    """

    cls_uuid: str = "view_angle"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._goalsensoruuid = getattr(self._config,"goalsensoruuid")

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [Success.cls_uuid]
        )
        self._metric = 180.
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        if ep_success:
            goal_sensor = task.sensor_suite.sensors[
                self._goalsensoruuid
            ]
            if hasattr(goal_sensor, 'get_goal_views'):
                goal_views = [
                    quaternion_from_coeff(v)
                    for v in goal_sensor.get_goal_views()
                ]
                agent_view = self._sim.get_agent_state().rotation
                dist_to_view = [
                    angle_between_quaternions(agent_view, qk)
                    for qk in goal_views
                ]
                dist_to_view = np.abs(np.array(dist_to_view)).min()
                self._metric = np.rad2deg(dist_to_view)
            else:
                self._metric = 180.0
        else:
            self._metric = 180.0


@registry.register_measure
class DistanceToView(Measure):
    r"""
    """

    cls_uuid: str = "distance_to_view"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._goalsensoruuid = getattr(self._config,"goalsensoruuid")
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):

        succ_d = getattr(task._config.success, "success_distance", 0)
        dist_to_goal = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        dist_to_view = np.pi
        goal_sensor = task.sensor_suite.sensors[
            self._goalsensoruuid
        ]
        if dist_to_goal <= succ_d and hasattr(goal_sensor, 'get_goal_views'):
            goal_views = [
                quaternion_from_coeff(v)
                for v in goal_sensor.get_goal_views()
            ]
            agent_view = self._sim.get_agent_state().rotation
            dist_to_view = [
                angle_between_quaternions(agent_view, qk)
                for qk in goal_views
            ]
            dist_to_view = np.abs(np.array(dist_to_view)).min()

        self._metric = dist_to_goal + dist_to_view


@registry.register_measure
class ViewMatch(Measure):
    r"""
    """

    cls_uuid: str = "view_match"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._goalsensoruuid = getattr(self._config,"goalsensoruuid")
        self._view_weight = getattr(self._config, "view_weight", 0.5)
        self._angle_threshold = np.deg2rad(self._config.angle_threshold)
        assert self._view_weight >= 0 and self._view_weight <= 1., "VIEW_WEIGHT has to be in [0, 1]"
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
        )
        self._metric = 0.
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()

        succ_d = getattr(task._config, "success_distance", 0.)
        dist_to_goal = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        if dist_to_goal <= succ_d:
            goal_sensor = task.sensor_suite.sensors[
                self._goalsensoruuid
            ]
            self._metric = 1 - self._view_weight
            if hasattr(goal_sensor, 'get_goal_views'):
                goal_views = [
                    quaternion_from_coeff(v)
                    for v in goal_sensor.get_goal_views()
                ]
                agent_view = self._sim.get_agent_state().rotation
                dist_to_view = [
                    angle_between_quaternions(agent_view, qk)
                    for qk in goal_views
                ]
                dist_to_view = np.abs(np.array(dist_to_view)).min()
                if dist_to_view <= self._angle_threshold:
                    self._metric += self._view_weight

            else:
                self._metric = 1.0
        else:
            self._metric = 0.0

        self._metric = ep_success * self._metric
