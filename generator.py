import abc
import json
import math
import random
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Set, Type, Optional

import cv2
import numpy as np

from random_f import TruncatedNormalSampler


def are_close(a, b, tolerance=0.01):
    return abs(a - b) < tolerance


def _to_point(value) -> Tuple[float, float]:
    return float(value[0]), float(value[1])


class Element(abc.ABC):

    def draw(self, img: np.ndarray, **kwargs) -> None:
        self.move_and_scale_to(img.shape[1] - 1, img.shape[0] - 1) \
            ._draw(img, **kwargs)

    def move_and_scale_to(self, max_x, max_y, min_x=0, min_y=0) -> 'Element':
        return self.scale_xy(max_x - min_x, max_y - min_y).move(min_x, min_y)

    def scale(self, factor: float) -> 'Element':
        return self.scale_xy(factor, factor)

    @property
    def refs(self) -> Set[str]:
        return set()

    @property
    def center(self) -> Tuple[float, float]:
        return (-1, -1)

    @abc.abstractmethod
    def _draw(self, img: np.ndarray, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def move(self, dx: float, dy: float) -> 'Element':
        pass

    @abc.abstractmethod
    def scale_xy(self, factor_x: float, factor_y: float) -> 'Element':
        pass

    @abc.abstractmethod
    def mirror_x(self) -> 'Element':
        pass

    @abc.abstractmethod
    def mirror_y(self) -> 'Element':
        pass

    @abc.abstractmethod
    def rotate(self, angle: float, center: Tuple[float, float] = (0.5, 0.5)) -> 'Element':
        pass

    @property
    def coordinates_params(self) -> List[Tuple[float, float]]:
        result = self._coordinates_params
        assert len(result) <= self.max_coordinates_params()
        return result

    @property
    def coordinates_params_flat(self) -> List[Tuple[float, float]]:
        for coordinates in self.coordinates_params:
            for coordinate in coordinates:
                yield coordinate

    @property
    def continuous_params(self) -> List[float]:
        result = self._continuous_params
        assert len(result) <= self.max_continuous_params()
        return result

    @property
    def discrete_params(self) -> List[str]:
        result = self._discrete_params
        assert len(result) <= self.max_discrete_params()
        return result

    @property
    @abc.abstractmethod
    def _coordinates_params(self) -> List[Tuple[float, float]]:
        pass

    @property
    @abc.abstractmethod
    def _continuous_params(self) -> List[float]:
        pass

    @property
    @abc.abstractmethod
    def _discrete_params(self) -> List[str]:
        pass

    @staticmethod
    @abc.abstractmethod
    def max_coordinates_params() -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def max_continuous_params() -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def max_discrete_params() -> int:
        pass

    @abc.abstractmethod
    def approx_equal(self, other: 'Element', resolution: float, **kwargs) -> bool:
        pass

    def serialize(self) -> dict:
        data = {}
        data['type'] = self.__class__.__name__
        if len(self.coordinates_params) > 0:
            data['coordinates_params'] = self.coordinates_params
        if len(self.continuous_params) > 0:
            data['continuous_params'] = self.continuous_params
        if len(self.discrete_params) > 0:
            data['discrete_params'] = self.discrete_params
        return data


def _rotate_point(point: Tuple[float, float], angle: float, center: Tuple[float, float]) -> Tuple[float, float]:
    rad = math.radians(angle)
    cos_angle, sin_angle = math.cos(rad), math.sin(rad)
    x, y = point
    cx, cy = center
    x -= cx
    y -= cy
    x_new = x * cos_angle - y * sin_angle
    y_new = x * sin_angle + y * cos_angle
    return x_new + cx, y_new + cy


@dataclass
class NoOp(Element):

    def _draw(self, img: np.ndarray, **kwargs) -> None:
        pass

    def move(self, dx: float, dy: float) -> 'Element':
        return self

    def scale_xy(self, factor_x: float, factor_y: float) -> 'Element':
        return self

    def rotate(self, angle: float, center: Tuple[float, float] = (0.5, 0.5)) -> 'Element':
        return self

    def mirror_x(self) -> 'Element':
        return self

    def mirror_y(self) -> 'Element':
        return self

    @property
    def _coordinates_params(self) -> List[Tuple[float, float]]:
        return []

    @property
    def _continuous_params(self) -> List[float]:
        return []

    @property
    def _discrete_params(self) -> List[str]:
        return []

    @staticmethod
    def max_coordinates_params() -> int:
        return 0

    @staticmethod
    def max_continuous_params() -> int:
        return 0

    @staticmethod
    def max_discrete_params() -> int:
        return 0

    def approx_equal(self, other: 'NoOp', resolution: float, **kwargs) -> bool:
        return isinstance(other, NoOp)


@dataclass
class FinishDrawing(NoOp):

    def to_xml(self) -> str:
        return "<FinishDrawing/>"

    def approx_equal(self, other: 'FinishDrawing', resolution: float, **kwargs) -> bool:
        return isinstance(other, FinishDrawing)


@dataclass
class PartLine(Element):
    """
    A straight line element, that is an edge or surface of the underlying 3D part.
    """
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    out_dir: Optional[Tuple[float, float]] = None

    def round_params(self, decimal_points: int=2) -> 'PartLine':
        return PartLine(
            (round(self.start_point[0], decimal_points), round(self.start_point[1], decimal_points)),
            (round(self.end_point[0], decimal_points), round(self.end_point[1], decimal_points)),
            None if self.out_dir is None else (round(self.out_dir[0], decimal_points), round(self.out_dir[1], decimal_points))
        )

    def _draw(self, img: np.ndarray, color, thickness, **kwargs) -> None:
        start_point = int(round(self.start_point[0])), int(round(self.start_point[1]))
        end_point = int(round(self.end_point[0])), int(round(self.end_point[1]))
        cv2.line(img, start_point, end_point, color, thickness)

    def move(self, dx, dy) -> 'PartLine':
        return PartLine(
            (self.start_point[0] + dx, self.start_point[1] + dy),
            (self.end_point[0] + dx, self.end_point[1] + dy),
            self.out_dir
        )

    @property
    def center(self) -> Tuple[float, float]:
        return (self.start_point[0] + self.end_point[0]) / 2, (self.start_point[1] + self.end_point[1]) / 2

    def scale_xy(self, factor_x: float, factor_y: float) -> 'PartLine':
        return PartLine(
            (self.start_point[0] * factor_x, self.start_point[1] * factor_y),
            (self.end_point[0] * factor_x, self.end_point[1] * factor_y),
            self.out_dir
        )

    def mirror_x(self) -> 'PartLine':
        return PartLine(
            (self.start_point[0], 1 - self.start_point[1]),
            (self.end_point[0], 1 - self.end_point[1]),
            (self.out_dir[0], -self.out_dir[1]) if self.out_dir is not None else None
        )

    def mirror_y(self) -> 'PartLine':
        return PartLine(
            (1 - self.start_point[0], self.start_point[1]),
            (1 - self.end_point[0], self.end_point[1]),
            (-self.out_dir[0], self.out_dir[1]) if self.out_dir is not None else None
        )

    def rotate(self, angle: float, center: Tuple[float, float] = (0.5, 0.5)) -> 'PartLine':
        return PartLine(
            _rotate_point(self.start_point, angle, center),
            _rotate_point(self.end_point, angle, center),
            _rotate_point(self.out_dir, angle, (0, 0)) if self.out_dir is not None else None
        )

    @staticmethod
    def fix_point_order(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        (x1, y1), (x2, y2) = p1, p2
        if not are_close(x1, x2):
            if x1 > x2:
                return p2, p1
        elif not are_close(y1, y2):
            if y1 > y2:
                return p2, p1
        return p1, p2

    @property
    def _coordinates_params(self) -> List[Tuple[float, float]]:
        return list(self.fix_point_order(self.start_point, self.end_point))

    @property
    def _continuous_params(self) -> List[float]:
        return []

    @property
    def _discrete_params(self) -> List[str]:
        return []

    @staticmethod
    def from_params(_: List[str], coordinate_params: List[Tuple[float, float]], continuous_params: List[float]) -> 'PartLine':
        return PartLine(
            coordinate_params[0],
            coordinate_params[1],
            None,
        )

    @staticmethod
    def max_coordinates_params() -> int:
        return 2

    @staticmethod
    def max_continuous_params() -> int:
        return 0

    @staticmethod
    def max_discrete_params() -> int:
        return 0

    @property
    def length(self) -> float:
        return math.hypot(self.start_point[0] - self.end_point[0], self.start_point[1] - self.end_point[1])

    def approx_equal(self, other: 'PartLine', resolution: float, **kwargs) -> bool:
        if not isinstance(other, PartLine):
            return False
        return np.allclose(self.start_point, other.start_point, atol=resolution) and np.allclose(self.end_point, other.end_point, atol=resolution)

    def distance(self, other: 'PartLine') -> float:
        return math.hypot(self.start_point[0] - other.start_point[0], self.start_point[1] - other.start_point[1]) + math.hypot(self.end_point[0] - other.end_point[0], self.end_point[1] - other.end_point[1])

    def to_xml(self) -> str:
        return f"<{self.__class__.__name__} x1={self.start_point[0]:.2f} y1={self.start_point[1]:.2f} x2={self.end_point[0]:.2f} y2={self.end_point[1]:.2f}/>"


@dataclass
class ConnectTwoElementsWithId(Element):
    id1: str
    id2: str

    def _draw(self, img: np.ndarray, **kwargs) -> None:
        pass  # connection is not visible

    def move(self, dx, dy) -> 'ConnectTwoElementsWithId':
        return self

    def scale_xy(self, factor_x: float, factor_y: float) -> 'ConnectTwoElementsWithId':
        return self

    def mirror_x(self) -> 'ConnectTwoElementsWithId':
        return self

    def mirror_y(self) -> 'ConnectTwoElementsWithId':
        return self

    def rotate(self, angle: float, center: Tuple[float, float] = (0.5, 0.5)) -> 'ConnectTwoElementsWithId':
        return self

    @property
    def _coordinates_params(self) -> List[Tuple[float, float]]:
        return []

    @property
    def _continuous_params(self) -> List[float]:
        return []

    @property
    def _discrete_params(self) -> List[str]:
        id1 = self.id1 if int(self.id1) < int(self.id2) else self.id2
        id2 = self.id2 if id1 == self.id1 else self.id1
        return [id1, id2]

    @property
    def refs(self) -> Set[str]:
        return {self.id1, self.id2}

    @staticmethod
    def max_coordinates_params() -> int:
        return 0

    @staticmethod
    def max_continuous_params() -> int:
        return 0

    @staticmethod
    def max_discrete_params() -> int:
        return 2

    def approx_equal(self, other: 'ConnectTwoElementsWithId', resolution: float, action_by_id, other_action_by_id) -> bool:
        if not isinstance(other, ConnectTwoElementsWithId):
            return False
        if self.id1 not in action_by_id or self.id2 not in action_by_id or other.id1 not in other_action_by_id or other.id2 not in other_action_by_id:
            return False
        a1 = action_by_id[self.id1]
        a2 = action_by_id[self.id2]
        other_a1 = other_action_by_id[other.id1]
        other_a2 = other_action_by_id[other.id2]
        return (a1.approx_equal(other_a1, resolution) and a2.approx_equal(other_a2, resolution)) \
            or (a1.approx_equal(other_a2, resolution) and a2.approx_equal(other_a1, resolution))

    @staticmethod
    def from_params(discrete_params: List[str], coordinate_params: List[Tuple[float, float]], continuous_params: List[float]) -> 'ConnectTwoElementsWithId':
        return ConnectTwoElementsWithId(discrete_params[0], discrete_params[1])


@dataclass
class PartLineWithId(Element):
    id: str
    part_line: PartLine

    def _draw(self, img: np.ndarray, **kwargs) -> None:
        return self.part_line._draw(img, **kwargs)

    def move(self, dx, dy) -> 'PartLineWithId':
        return PartLineWithId(
            self.id,
            self.part_line.move(dx, dy),
        )

    @property
    def center(self) -> Tuple[float, float]:
        return self.part_line.center

    def scale_xy(self, factor_x: float, factor_y: float) -> 'PartLineWithId':
        return PartLineWithId(
            self.id,
            self.part_line.scale_xy(factor_x, factor_y),
        )

    def mirror_x(self) -> 'PartLineWithId':
        return PartLineWithId(
            self.id,
            self.part_line.mirror_x(),
        )

    def mirror_y(self) -> 'PartLineWithId':
        return PartLineWithId(
            self.id,
            self.part_line.mirror_y(),
        )

    def rotate(self, angle: float, center: Tuple[float, float] = (0.5, 0.5)) -> 'PartLineWithId':
        return PartLineWithId(
            self.id,
            self.part_line.rotate(angle, center),
        )

    @property
    def _coordinates_params(self) -> List[Tuple[float, float]]:
        return self.part_line._coordinates_params

    @property
    def _continuous_params(self) -> List[float]:
        return self.part_line._continuous_params

    @property
    def _discrete_params(self) -> List[str]:
        return self.part_line._discrete_params

    @staticmethod
    def max_coordinates_params() -> int:
        return PartLine.max_coordinates_params()

    @staticmethod
    def max_continuous_params() -> int:
        return PartLine.max_continuous_params()

    @staticmethod
    def max_discrete_params() -> int:
        return PartLine.max_discrete_params()

    def approx_equal(self, other: 'PartLineWithId', resolution: float, **kwargs) -> bool:
        if not isinstance(other, PartLineWithId):
            return False
        return self.part_line.approx_equal(other.part_line, resolution)

    @staticmethod
    def from_params(discrete_params: List[str], coordinate_params: List[Tuple[float, float]], continuous_params: List[float], action_index: int) -> 'PartLineWithId':
        return PartLineWithId(
            str(action_index),
            PartLine.from_params(discrete_params[:-1], coordinate_params, continuous_params)
        )


@dataclass
class HelpLine(Element):
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    formative_only: bool = False

    @staticmethod
    def from_params(_: List[str], coordinates_params: List[Tuple[float, float]], continuous_params: List[float]) -> 'HelpLine':
        return HelpLine(
            coordinates_params[0],
            coordinates_params[1],
        )

    @property
    def center(self) -> Tuple[float, float]:
        return (self.start_point[0] + self.end_point[0]) / 2, (self.start_point[1] + self.end_point[1]) / 2

    def _draw(self, img: np.ndarray, color, thickness, **kwargs) -> None:
        start_point = int(round(self.start_point[0])), int(round(self.start_point[1]))
        end_point = int(round(self.end_point[0])), int(round(self.end_point[1]))
        cv2.line(img, start_point, end_point, color, thickness)

    def move(self, dx, dy) -> 'HelpLine':
        return HelpLine(
            (self.start_point[0] + dx, self.start_point[1] + dy),
            (self.end_point[0] + dx, self.end_point[1] + dy),
            formative_only=self.formative_only,
        )

    def scale_xy(self, factor_x: float, factor_y: float) -> 'HelpLine':
        return HelpLine(
            (self.start_point[0] * factor_x, self.start_point[1] * factor_y),
            (self.end_point[0] * factor_x, self.end_point[1] * factor_y),
            formative_only=self.formative_only,
        )

    def mirror_x(self) -> 'HelpLine':
        return HelpLine(
            (1 - self.start_point[0], self.start_point[1]),
            (1 - self.end_point[0], self.end_point[1]),
            formative_only=self.formative_only,
        )

    def mirror_y(self) -> 'HelpLine':
        return HelpLine(
            (self.start_point[0], 1 - self.start_point[1]),
            (self.end_point[0], 1 - self.end_point[1]),
            formative_only=self.formative_only,
        )

    def rotate(self, angle: float, center: Tuple[float, float] = (0.5, 0.5)) -> 'HelpLine':
        return HelpLine(
            _rotate_point(self.start_point, angle, center),
            _rotate_point(self.end_point, angle, center),
            formative_only=self.formative_only,
        )

    @property
    def _coordinates_params(self) -> List[Tuple[float, float]]:
        return list(PartLine.fix_point_order(self.start_point, self.end_point))

    @property
    def _continuous_params(self) -> List[float]:
        return []

    @property
    def _discrete_params(self) -> List[str]:
        return []

    @staticmethod
    def max_coordinates_params() -> int:
        return 2

    @staticmethod
    def max_continuous_params() -> int:
        return 0

    @staticmethod
    def max_discrete_params() -> int:
        return 0

    def approx_equal(self, other: 'HelpLine', resolution: float, **kwargs) -> bool:
        if not isinstance(other, HelpLine):
            return False
        return np.allclose(self.start_point, other.start_point, atol=resolution) and np.allclose(self.end_point, other.end_point, atol=resolution)


@dataclass
class BothSidedArrow(Element):
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    formative_only: bool = False

    @staticmethod
    def from_params(_: List[str], coordinates_params: List[Tuple[float, float]], continuous_params: List[float]) -> 'BothSidedArrow':
        return BothSidedArrow(
            coordinates_params[0],
            coordinates_params[1],
        )

    @property
    def center(self) -> Tuple[float, float]:
        return (self.start_point[0] + self.end_point[0]) / 2, (self.start_point[1] + self.end_point[1]) / 2

    def _draw(self, img: np.ndarray, color, thickness, arrow_type, **kwargs) -> None:
        start_point = int(round(self.start_point[0])), int(round(self.start_point[1]))
        end_point = int(round(self.end_point[0])), int(round(self.end_point[1]))

        if arrow_type == 'ArrowedLine':
            self._draw_arrowed_line(img, start_point, end_point, color, thickness)
        elif arrow_type == 'FilledTriangleArrow':
            self._draw_filled_triangle(img, start_point, end_point, color, thickness)
        elif arrow_type == 'NonFilledTriangleArrow':
            self._draw_non_filled_triangle(img, start_point, end_point, color, thickness)

    def _draw_arrowed_line(self, img, start_point, end_point, color, thickness):

        cv2.line(img, start_point, end_point, color, thickness)

        direction = np.array(end_point) - np.array(start_point)

        if np.linalg.norm(direction) == 0:
            return

        direction_norm = direction / np.linalg.norm(direction)

        arrowhead_length = 10
        arrowhead_angle = np.pi / 6

        end_arrowhead_point1 = np.array(end_point) - arrowhead_length * (
                np.cos(arrowhead_angle) * direction_norm + np.sin(arrowhead_angle) * np.array(
            [-direction_norm[1], direction_norm[0]]))
        end_arrowhead_point2 = np.array(end_point) - arrowhead_length * (
                np.cos(arrowhead_angle) * direction_norm - np.sin(arrowhead_angle) * np.array(
            [-direction_norm[1], direction_norm[0]]))

        cv2.line(img, tuple(end_point), tuple(map(int, end_arrowhead_point1)), color, thickness)
        cv2.line(img, tuple(end_point), tuple(map(int, end_arrowhead_point2)), color, thickness)

        start_arrowhead_point1 = np.array(start_point) + arrowhead_length * (
                np.cos(arrowhead_angle) * direction_norm - np.sin(arrowhead_angle) * np.array(
            [-direction_norm[1], direction_norm[0]]))
        start_arrowhead_point2 = np.array(start_point) + arrowhead_length * (
                np.cos(arrowhead_angle) * direction_norm + np.sin(arrowhead_angle) * np.array(
            [-direction_norm[1], direction_norm[0]]))

        cv2.line(img, tuple(start_point), tuple(map(int, start_arrowhead_point1)), color, thickness)
        cv2.line(img, tuple(start_point), tuple(map(int, start_arrowhead_point2)), color, thickness)

    @staticmethod
    def _draw_filled_triangle(img, start_point, end_point, color, thickness):

        cv2.line(img, start_point, end_point, color, thickness)

        arrow_direction = np.array(end_point) - np.array(start_point)

        if np.linalg.norm(arrow_direction) == 0:
            return

        arrow_direction = arrow_direction / np.linalg.norm(arrow_direction)

        arrowhead_length = 10 / 2
        arrowhead_width = 5 / 2

        arrowhead_point1 = np.array(end_point) - arrow_direction * arrowhead_length + np.array(
            [-arrow_direction[1], arrow_direction[0]]) * arrowhead_width
        arrowhead_point2 = np.array(end_point) - arrow_direction * arrowhead_length - np.array(
            [-arrow_direction[1], arrow_direction[0]]) * arrowhead_width

        arrowhead_point3 = np.array(start_point) + arrow_direction * arrowhead_length + np.array(
            [-arrow_direction[1], arrow_direction[0]]) * arrowhead_width
        arrowhead_point4 = np.array(start_point) + arrow_direction * arrowhead_length - np.array(
            [-arrow_direction[1], arrow_direction[0]]) * arrowhead_width

        triangle_cnt = np.array([end_point, tuple(map(int, arrowhead_point1)), tuple(map(int, arrowhead_point2))])
        triangle_cnt2 = np.array([start_point, tuple(map(int, arrowhead_point3)), tuple(map(int, arrowhead_point4))])

        cv2.fillPoly(img, [triangle_cnt], color)
        cv2.fillPoly(img, [triangle_cnt2], color)
        cv2.fillPoly(img, [triangle_cnt2], color)

    def _draw_non_filled_triangle(self, img, start_point, end_point, color, thickness):
        cv2.line(img, start_point, end_point, color, thickness)

        arrow_direction = np.array(end_point) - np.array(start_point)
        arrow_direction = arrow_direction / np.linalg.norm(arrow_direction)

        arrowhead_length = 10
        arrowhead_width = 5

        arrowhead_point1 = np.array(end_point) - arrow_direction * arrowhead_length + np.array(
            [-arrow_direction[1], arrow_direction[0]]) * arrowhead_width
        arrowhead_point2 = np.array(end_point) - arrow_direction * arrowhead_length - np.array(
            [-arrow_direction[1], arrow_direction[0]]) * arrowhead_width

        arrowhead_point3 = np.array(start_point) + arrow_direction * arrowhead_length + np.array(
            [-arrow_direction[1], arrow_direction[0]]) * arrowhead_width
        arrowhead_point4 = np.array(start_point) + arrow_direction * arrowhead_length - np.array(
            [-arrow_direction[1], arrow_direction[0]]) * arrowhead_width

        cv2.line(img, tuple(map(int, arrowhead_point1)), end_point, color, thickness)
        cv2.line(img, tuple(map(int, arrowhead_point2)), end_point, color, thickness)
        cv2.line(img, tuple(map(int, arrowhead_point1)), tuple(map(int, arrowhead_point2)), color, thickness)

        cv2.line(img, tuple(map(int, arrowhead_point3)), start_point, color, thickness)
        cv2.line(img, tuple(map(int, arrowhead_point4)), start_point, color, thickness)
        cv2.line(img, tuple(map(int, arrowhead_point3)), tuple(map(int, arrowhead_point4)), color, thickness)

    def move(self, dx: float, dy: float) -> 'BothSidedArrow':
        new_start = (self.start_point[0] + dx, self.start_point[1] + dy)
        new_end = (self.end_point[0] + dx, self.end_point[1] + dy)
        return BothSidedArrow(new_start, new_end, formative_only=self.formative_only)

    def scale_xy(self, factor_x: float, factor_y: float) -> 'BothSidedArrow':
        new_start = (self.start_point[0] * factor_x, self.start_point[1] * factor_y)
        new_end = (self.end_point[0] * factor_x, self.end_point[1] * factor_y)
        return BothSidedArrow(new_start, new_end, formative_only=self.formative_only)

    def mirror_x(self) -> 'BothSidedArrow':
        mirrored_start_point = (self.start_point[0], 1 - self.start_point[1])
        mirrored_end_point = (self.end_point[0], 1 - self.end_point[1])
        return BothSidedArrow(mirrored_start_point, mirrored_end_point, formative_only=self.formative_only)

    def mirror_y(self) -> 'BothSidedArrow':
        mirrored_start_point = (1 - self.start_point[0], self.start_point[1])
        mirrored_end_point = (1 - self.end_point[0], self.end_point[1])
        return BothSidedArrow(mirrored_start_point, mirrored_end_point, formative_only=self.formative_only)

    def rotate(self, angle: float, center: Tuple[float, float] = (0.5, 0.5)) -> 'BothSidedArrow':
        rotated_start_point = _rotate_point(self.start_point, angle, center)
        rotated_end_point = _rotate_point(self.end_point, angle, center)
        return BothSidedArrow(rotated_start_point, rotated_end_point, formative_only=self.formative_only)

    @property
    def _coordinates_params(self) -> List[Tuple[float, float]]:
        return list(PartLine.fix_point_order(self.start_point, self.end_point))

    @property
    def _continuous_params(self) -> List[float]:
        return []

    @property
    def _discrete_params(self) -> List[str]:
        return []

    @staticmethod
    def max_coordinates_params() -> int:
        return 2

    @staticmethod
    def max_continuous_params() -> int:
        return 0

    @staticmethod
    def max_discrete_params() -> int:
        return 0

    def approx_equal(self, other: 'BothSidedArrow', resolution: float, **kwargs) -> bool:
        if not isinstance(other, BothSidedArrow):
            return False
        return np.allclose(self.start_point, other.start_point, atol=resolution) and np.allclose(self.end_point, other.end_point, atol=resolution)


@dataclass
class Circle(Element):
    _center: Tuple[float, float]
    radius: float

    @property
    def center(self) -> Tuple[float, float]:
        return self._center

    @property
    def x(self) -> float:
        return self.center[0]

    @property
    def y(self) -> float:
        return self.center[1]

    def round_params(self, decimal_points: int=2) -> 'Circle':
        return Circle(
            (round(self.center[0], decimal_points), round(self.center[1], decimal_points)),
            round(self.radius, decimal_points)
        )

    def _draw(self, img: np.ndarray, color, thickness, **kwargs) -> None:
        center = int(round(self.center[0])), int(round(self.center[1]))
        radius = int(round(self.radius))
        if radius <= 0:
            return
        cv2.circle(img, center, radius, color, thickness)

    def move(self, dx, dy) -> 'Circle':
        return Circle(
            (self.center[0] + dx, self.center[1] + dy),
            self.radius,
        )

    def scale_xy(self, factor_x: float, factor_y: float, radius_factor: float = None) -> 'Circle':
        if radius_factor is None:
            radius_factor = (factor_x + factor_y) / 2
        return Circle(
            (self.center[0] * factor_x, self.center[1] * factor_y),
            self.radius * radius_factor,
        )

    @property
    def _coordinates_params(self) -> List[Tuple[float, float]]:
        return [
            (self.center[0] - self.radius, self.center[1]), 
            (self.center[0] + self.radius, self.center[1])   
        ]

    @property
    def _continuous_params(self) -> List[float]:
        return []

    @property
    def _discrete_params(self) -> List[str]:
        return []

    @staticmethod
    def from_params(_: List[str], coordinates_params: List[Tuple[float, float]], continuous_params: List[float]) -> 'Circle':
        radius = abs(coordinates_params[0][0] - coordinates_params[1][0]) / 2
        center = (coordinates_params[0][0] + coordinates_params[1][0]) / 2, (coordinates_params[0][1] + coordinates_params[1][1]) / 2
        return Circle(
            center,
            radius,
        )

    @staticmethod
    def max_coordinates_params() -> int:
        return 2

    @staticmethod
    def max_continuous_params() -> int:
        return 0

    @staticmethod
    def max_discrete_params() -> int:
        return 0

    def mirror_x(self) -> 'Circle':
        return Circle(
            (1 - self.center[0], self.center[1]),
            self.radius
        )

    def mirror_y(self) -> 'Circle':
        return Circle(
            (self.center[0], 1 - self.center[1]),
            self.radius
        )

    def rotate(self, angle: float, center: Tuple[float, float] = (0.5, 0.5)) -> 'Circle':
        return Circle(
            _rotate_point(self.center, angle, center),
            self.radius
        )

    def approx_equal(self, other: 'Circle', resolution: float, **kwargs) -> bool:
        if not isinstance(other, Circle):
            return False
        return np.allclose(self.center, other.center, atol=resolution) and np.allclose(self.radius, other.radius, atol=resolution)

    def distance(self, other: 'Circle') -> float:
        return np.linalg.norm(np.array(self.center) - np.array(other.center)) + abs(self.radius - other.radius)


@dataclass
class AnnotationText(Element):
    point: Tuple[float, float]
    text_angle: float

    @property
    def center(self) -> Tuple[float, float]:
        return self.point

    def _draw(self, img: np.ndarray, color, thickness, font, font_size, **kwargs) -> None:
        random_text = str(int(np.random.uniform(0, 100)))
        text = self.draw_text(random_text, font, font_size=font_size)
        text = self.rotate_image(text, self.text_angle)

        height, width = img.shape[0], img.shape[1]
        text_height, text_width = text.shape[0], text.shape[1]

        x, y = int(round(self.point[0])), int(round(self.point[1]))

        start_x = x - text_width // 2
        start_y = y - text_height // 2
        end_x = start_x + text_width
        end_y = start_y + text_height
        target_start_x, target_start_y = max(start_x, 0), max(start_y, 0)
        target_end_x, target_end_y = min(end_x, width), min(end_y, height)
        text_start_x, text_start_y = max(-start_x, 0), max(-start_y, 0)
        text_end_x = min(text_width, text_width - max(end_x - width, 0))
        text_end_y = min(text_height, text_height - max(end_y - height, 0))

        target_end_x, target_end_y = max(target_end_x, target_start_x), max(target_end_y, target_start_y)
        text_end_x, text_end_y = max(text_end_x, text_start_x), max(text_end_y, text_start_y)

        try:
            if len(img.shape) == 2:
                img[target_start_y:target_end_y, target_start_x:target_end_x] += text[text_start_y:text_end_y, text_start_x:text_end_x]
            else:
                img[target_start_y:target_end_y, target_start_x:target_end_x, 0] += text[text_start_y:text_end_y, text_start_x:text_end_x]
            img[img > 255] = 255
        except Exception as e:
            print("ERROR IN DRAWING TEXT", e)

    @staticmethod
    def from_params(discrete_params: List[str], coordinates_params: List[Tuple[float, float]], continuous_params: List[float]) -> 'AnnotationText':
        angle_class = discrete_params[0]
        text_angle = int(angle_class.split("_")[-1]) * 90.0
        return AnnotationText(
            coordinates_params[0],
            text_angle,
        )

    @staticmethod
    def rotate_image(image, angle):
        normalized_angle = angle % 360
        if np.isclose(normalized_angle % 90, 0):
            quarter_turns = int(round(normalized_angle / 90)) % 4
            if quarter_turns == 0:
                return image
            return np.rot90(image, k=quarter_turns)
        import imutils
        return imutils.rotate_bound(image, angle)

    @staticmethod
    def draw_text(text: str, font, font_size):
        from PIL import Image, ImageDraw
        scratch = Image.new("L", (1, 1), 0)
        scratch_draw = ImageDraw.Draw(scratch)
        left, top, right, bottom = scratch_draw.textbbox((0, 0), text, font=font)
        padding = max(2, int(np.ceil(font_size * 0.35)))
        width = max(1, int(np.ceil(right - left)) + 2 * padding)
        height = max(1, int(np.ceil(bottom - top)) + 2 * padding)

        text_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(text_img)
        draw.text((padding - left, padding - top), text, font=font, fill=255)
        text_np = np.array(text_img)
        non_zero = np.argwhere(text_np > 0)
        if non_zero.size == 0:
            return text_np
        min_y, min_x = non_zero.min(axis=0)
        max_y, max_x = non_zero.max(axis=0)
        return text_np[min_y:max_y + 1, min_x:max_x + 1]

    def move(self, dx, dy) -> 'AnnotationText':
        return AnnotationText((
            self.point[0] + dx, self.point[1] + dy
        ), self.text_angle)

    def scale_xy(self, factor_x: float, factor_y: float) -> 'AnnotationText':
        return AnnotationText((
            self.point[0] * factor_x, self.point[1] * factor_y
        ), self.text_angle)

    def mirror_x(self) -> 'AnnotationText':
        return self

    def mirror_y(self) -> 'AnnotationText':
        return self

    def rotate(self, angle: float, center: Tuple[float, float] = (0.5, 0.5)) -> 'AnnotationText':
        return self

    @property
    def _coordinates_params(self) -> List[Tuple[float, float]]:
        return [self.point]

    @property
    def _continuous_params(self) -> List[float]:
        return []

    @property
    def _discrete_params(self) -> List[str]:
        angle_class = int(self.text_angle / 90) % 4
        return [f"angle_{angle_class}"]

    @staticmethod
    def max_coordinates_params() -> int:
        return 1

    @staticmethod
    def max_continuous_params() -> int:
        return 0

    @staticmethod
    def max_discrete_params() -> int:
        return 1

    def approx_equal(self, other: 'AnnotationText', resolution: float, **kwargs) -> bool:
        if not isinstance(other, AnnotationText):
            return False
        return np.allclose(self.point, other.point, atol=resolution)


@dataclass
class AnnotationTextRefId(Element):
    ref_id: str
    annotation_text: AnnotationText

    @property
    def center(self) -> Tuple[float, float]:
        return self.annotation_text.center

    def _draw(self, img: np.ndarray, **kwargs) -> None:
        return self.annotation_text._draw(img, **kwargs)

    @staticmethod
    def from_params(discrete_params: List[str], coordinates_params: List[Tuple[float, float]], continuous_params: List[float]) -> 'AnnotationTextRefId':
        return AnnotationTextRefId(
            discrete_params[-1],
            AnnotationText.from_params(discrete_params[:-1], coordinates_params, continuous_params)
        )

    def move(self, dx, dy) -> 'AnnotationTextRefId':
        return AnnotationTextRefId(
            self.ref_id,
            self.annotation_text.move(dx, dy),
        )

    def scale_xy(self, factor_x: float, factor_y: float) -> 'AnnotationTextRefId':
        return AnnotationTextRefId(
            self.ref_id,
            self.annotation_text.scale_xy(factor_x, factor_y),
        )

    def mirror_x(self) -> 'Element':
        return AnnotationTextRefId(self.ref_id, self.annotation_text.mirror_x())

    def mirror_y(self) -> 'Element':
        return AnnotationTextRefId(self.ref_id, self.annotation_text.mirror_y())

    def rotate(self, angle: float, center: Tuple[float, float] = (0.5, 0.5)) -> 'Element':
        return AnnotationTextRefId(self.ref_id, self.annotation_text.rotate(angle, center))

    @property
    def refs(self) -> Set[str]:
        return {self.ref_id}

    @property
    def _coordinates_params(self) -> List[Tuple[float, float]]:
        return self.annotation_text._coordinates_params

    @property
    def _continuous_params(self) -> List[float]:
        return self.annotation_text._continuous_params

    @property
    def _discrete_params(self) -> List[str]:
        return self.annotation_text._discrete_params + [self.ref_id]

    @staticmethod
    def max_coordinates_params() -> int:
        return AnnotationText.max_coordinates_params()

    @staticmethod
    def max_continuous_params() -> int:
        return AnnotationText.max_continuous_params()

    @staticmethod
    def max_discrete_params() -> int:
        return AnnotationText.max_discrete_params() + 1

    def approx_equal(self, other: 'AnnotationTextRefId', resolution: float, action_by_id, other_action_by_id, **kwargs) -> bool:
        if not isinstance(other, AnnotationTextRefId):
            return False
        if self.ref_id not in action_by_id or other.ref_id not in other_action_by_id:
            return False
        ref_a = action_by_id[self.ref_id]
        other_ref_a = other_action_by_id[other.ref_id]
        return ref_a.approx_equal(other_ref_a, resolution) and self.annotation_text.approx_equal(other.annotation_text, resolution)


class Generator(abc.ABC):
    """
    A generator for elements.

    A generator generates a list of elements and is at the heart of the generator framework.
    Having a clear interface allows:
    * combinations of generators to create more complex generators
    * decoration of generators to extend/modify their behavior
    """

    @abc.abstractmethod
    def get_actions(self) -> List[Element]:
        pass

    @abc.abstractmethod
    def action_types(self) -> Set[Type[Element]]:
        pass

    @property
    @abc.abstractmethod
    def n_actions(self) -> int:
        pass

    @property
    def max_continuous_params_per_action(self) -> int:
        return max(action_type.max_continuous_params() for action_type in self.action_types())

    @property
    def max_coordinates_params_per_action(self) -> int:
        return max(action_type.max_coordinates_params() for action_type in self.action_types())

    @property
    def max_discrete_params_per_action(self) -> int:
        return max(action_type.max_discrete_params() for action_type in self.action_types())

    @property
    def max_total_continuous_params(self) -> int:
        return self.max_continuous_params_per_action * self.n_actions

    @property
    def max_total_discrete_params(self) -> int:
        return self.max_discrete_params_per_action * self.n_actions


class HorizontalLineGenerator(Generator):
    """
    A generator that generates 1 horizontal line with coordinates between 0 and 1.
    """

    def __init__(self, min_length: float = 0.1, out_dir=None):
        self.min_length = min_length
        self.out_dir = out_dir

    def get_actions(self) -> List[PartLine]:
        length = random.uniform(self.min_length, 1.0)
        x1, y1 = random.uniform(0.0, 1.0 - length), random.uniform(0.0, 1.0)
        x2, y2 = x1 + length, y1
        return [
            PartLine((x1, y1), (x2, y2), self.out_dir)
        ]

    def action_types(self) -> Set[Type[Element]]:
        return {PartLine}

    @property
    def n_actions(self) -> int:
        return 1


class RadialLineGenerator(Generator):
    """
    A generator that generates 1 random line with start point (0, 0).
    """

    def get_actions(self) -> List[Element]:
        length = random.uniform(0.0, 1.0)
        angle = np.random.uniform(0, np.pi / 2)
        x1, y1 = 0.0, 0.0
        x2, y2 = x1 + length * np.cos(angle), y1 + length * np.sin(angle)
        return [
            PartLine((x1, y1), (x2, y2), None)
        ]

    def action_types(self) -> Set[Type[Element]]:
        return {PartLine}

    @property
    def n_actions(self) -> int:
        return 1


class AngularSortedPolygonGenerator(Generator):
    """
    A generator that generates a random polygon.
    It generates a random number of points and connects them in angular order by Line.
    """

    def __init__(self, num_points: int):
        self.num_points = num_points

    def get_actions(self) -> List[Element]:
        points = np.random.rand(self.num_points, 2)

        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]

        lines = []
        for i in range(len(sorted_points)):
            start_point = sorted_points[i]
            end_point = sorted_points[(i + 1) % len(sorted_points)]
            lines.append(PartLine((start_point[0], start_point[1]), (end_point[0], end_point[1]), None))

        return lines

    def action_types(self) -> Set[Type[Element]]:
        return {PartLine}

    @property
    def n_actions(self) -> int:
        return self.num_points


class ConvexHullPolygonGenerator(Generator):

    """
    A generator that generates a random convex polygon.

    It generates a random number of points and computes and draws the convex hull of these points using Line.
    """

    def __init__(self, num_points: int):
        self.num_points = num_points

    def get_actions(self) -> List[PartLine]:
        from scipy.spatial import ConvexHull

        # Generate random points
        points = np.random.rand(self.num_points, 2)

        # Compute the convex hull
        hull = ConvexHull(points)

        # Extract the vertices of the convex hull in the correct order
        hull_points = points[hull.vertices]

        lines = []
        num_hull_points = len(hull_points)
        for i in range(num_hull_points):
            start_point = hull_points[i]
            end_point = hull_points[(i + 1) % num_hull_points]
            lines.append(PartLine((start_point[0], start_point[1]), (end_point[0], end_point[1]), out_dir=None))

        return lines

    def action_types(self) -> Set[Type[Element]]:
        return {PartLine}

    @property
    def n_actions(self) -> int:
        return self.num_points


def angle_between(l1: PartLine, l2: PartLine):
    s1, e1 = np.array(l1.start_point), np.array(l1.end_point)
    s2, e2 = np.array(l2.start_point), np.array(l2.end_point)
    if np.equal(s1, e2).all():
        s2, e2 = e2, s2
    elif np.equal(s2, e1).all():
        s1, e1 = e1, s1
    assert np.equal(s1, s2).all()
    v1, v2 = s1 - e1, s2 - e2
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return min([angle, 2*np.pi - angle])  # Smaller Angle is inside the convex hull


class ConvexHullPolygonGenerator2(Generator):

    """
    A generator that generates a random convex polygon.

    It generates a random number of points and computes and draws the convex hull of these points using Line.
    """

    def __init__(self, num_points: int, min_inner_angle: float):
        self.generator = ConvexHullPolygonGenerator(num_points=num_points)
        self.min_inner_angle = min_inner_angle

    def get_actions(self) -> List[Element]:
        def angles_between_lines(lines: List[PartLine]):
            line_pairs = list(zip(lines, lines[1:] + [lines[0]]))
            return (angle_between(l1, l2) for l1, l2 in line_pairs)
        lines = self.generator.get_actions()
        while max(angles_between_lines(lines)) > math.radians(self.min_inner_angle):
            lines = self.generator.get_actions()
        return lines

    def action_types(self) -> Set[Type[Element]]:
        return {PartLine}

    @property
    def n_actions(self) -> int:
        return self.generator.n_actions


class CentricCircleGenerator(Generator):
    """
    A generator that generates 1 random circle with its center at (0.5, 0.5).
    """

    def get_actions(self) -> List[Circle]:
        radius = random.uniform(0.1, 0.5)
        return [Circle((0.5, 0.5), radius)]

    def action_types(self) -> Set[Type[Element]]:
        return {Circle}

    @property
    def n_actions(self) -> int:
        return 1

class RectangleGenerator(Generator):

    def __init__(self, min_size: float = 0.2, max_size: float = 0.8):
        self.min_size = min_size
        self.max_size = max_size

    def get_actions(self) -> List[Element]:
        width = random.uniform(self.min_size, self.max_size)
        height = random.uniform(self.min_size, self.max_size)

        top_left_x = random.uniform(0, 1 - width)
        top_left_y = random.uniform(0, 1 - height)

        top_right = (top_left_x + width, top_left_y)
        bottom_left = (top_left_x, top_left_y + height)
        bottom_right = (top_left_x + width, top_left_y + height)

        return [
            PartLine((top_left_x, top_left_y), top_right, (0, -1)),
            PartLine(top_right, bottom_right, (1, 0)),
            PartLine(bottom_right, bottom_left, (0, 1)),
            PartLine(bottom_left, (top_left_x, top_left_y), (-1, 0))
        ]

    def action_types(self) -> Set[Type[Element]]:
        return {PartLine}

    @property
    def n_actions(self) -> int:
        return 4


class RectangleWithCirclesGenerator(Generator):
    def __init__(self, num_circles: int, min_size: float = 0.05, max_size: float = 1.0):
        self.min_size = min_size
        self.max_size = max_size
        self.num_circles = num_circles

    def get_actions(self) -> List[Element]:
        width = random.uniform(self.min_size, self.max_size)
        height = random.uniform(self.min_size, self.max_size)
        top_left_x = random.uniform(0, 1 - width)
        top_left_y = random.uniform(0, 1 - height)

        top_right = (top_left_x + width, top_left_y)
        bottom_left = (top_left_x, top_left_y + height)
        bottom_right = (top_left_x + width, top_left_y + height)

        rectangle_lines = [
            PartLine((top_left_x, top_left_y), top_right, (0, -1)),
            PartLine(top_right, bottom_right, (1, 0)),
            PartLine(bottom_right, bottom_left, (0, 1)),
            PartLine(bottom_left, (top_left_x, top_left_y), (-1, 0))
        ]

        circles = self.generate_non_overlapping_circles(top_left_x, top_left_y, width, height, self.num_circles)

        return rectangle_lines + circles

    def generate_non_overlapping_circles(self, x, y, width, height, num_circles):
        circles = []
        attempts = 0
        max_attempts = 100

        while len(circles) < num_circles and attempts < max_attempts:
            radius = random.uniform(0.05, min(width, height) / 4)
            circle_x = random.uniform(x + radius, x + width - radius)
            circle_y = random.uniform(y + radius, y + height - radius)
            new_circle = Circle((circle_x, circle_y), radius)

            overlap = any(self.circles_overlap(new_circle, existing_circle) for existing_circle in circles)
            if not overlap:
                circles.append(new_circle)
            attempts += 1

        return circles

    def circles_overlap(self, circle1, circle2):
        dx = circle1.x - circle2.x
        dy = circle1.y - circle2.y
        radii_sum = circle1.radius + circle2.radius
        return dx * dx + dy * dy < radii_sum * radii_sum

    def action_types(self) -> Set[Type[Element]]:
        return {PartLine, Circle}

    @property
    def n_actions(self) -> int:
        return 4 + self.num_circles


class GeneratorDecorator(Generator, ABC):

    """
    A base decorator for generators (Decorator Pattern).

    This class is meant to be subclassed by decorators that want to modify a specific behavior of a generator, e.g. decorate get_actions.
    """

    def __init__(self, generator: Generator):
        self.generator = generator

    def get_actions(self) -> List[Element]:
        return self.generator.get_actions()

    def action_types(self) -> Set[Type[Element]]:
        return self.generator.action_types()

    @property
    def n_actions(self) -> int:
        return self.generator.n_actions


class HorizontalCircleGenerator(Generator):
    """
    A generator that generates num_circles horizontal circles fully visible between 0 and 1.
    """

    def __init__(self, num_circles: int):
        self.num_circles = num_circles

    @staticmethod
    def _normalize_circles(circles: List[Circle]) -> List[Circle]:
        min_x = min(circle.x - circle.radius for circle in circles)
        max_x = max(circle.x + circle.radius for circle in circles)
        scale = 1 / (max_x - min_x)
        return list(
            circle.move(-min_x, 0).scale_xy(scale, 1, radius_factor=scale)
            for circle in circles
        )

    def get_actions(self) -> List[Element]:
        circles = []
        x, y = 0, 0.5
        distance_between_circles = 0.1
        for i in range(self.num_circles):
            radius = random.uniform(0.1, 0.5)
            circle_center = (x + radius, y)
            circles.append(Circle(circle_center, radius))
            x += radius * 2 + distance_between_circles
        return self._normalize_circles(circles)

    def action_types(self) -> Set[Type[Element]]:
        return {Circle}

    @property
    def n_actions(self) -> int:
        return self.num_circles


class LShapeGenerator(Generator):
    """
    A generator that generates an L shape using 6 PartLine.
    """

    def __init__(self, min_length: float = 0.1):
        assert 0.0 < min_length <= 0.5, "min_length must be in (0, 0.5]"
        self.min_length = min_length

    def get_actions(self) -> List[PartLine]:
        x, y = 0.0, 0.0
        w1, h1 = random.uniform(self.min_length, 1.0 - self.min_length), random.uniform(self.min_length, 1.0 - self.min_length)
        w2, h2 = 1.0 - (x + w1), 1.0 - (y + h1)
        return [
            PartLine((x, y), (x, y + h1), (-1, 0)),  # Left
            PartLine((x, y + h1), (x + w1, y + h1), (0, 1)),  # Bottom
            PartLine((x + w1, y + h1), (x + w1, y + h1 + h2), (-1, 0)),  # Left
            PartLine((x + w1, y + h1 + h2), (x + w1 + w2, y + h1 + h2), (0, 1)),  # Bottom
            PartLine((x + w1 + w2, y + h1 + h2), (x + w1 + w2, y), (1, 0)),  # Right
            PartLine((x + w1 + w2, y), (x, y), (0, -1))  # Top
        ]

    def action_types(self) -> Set[Type[Element]]:
        return {PartLine}

    @property
    def n_actions(self) -> int:
        return 6


class LShapeOutlineGenerator(Generator):
    """
    A generator that generates an L shape using 6 PartLine.
    """

    def __init__(self, min_length: float = 0.1):
        assert 0.0 < min_length <= 0.5, "min_length must be in (0, 0.5]"
        self.min_length = min_length

    def get_actions(self) -> List[Element]:
        x, y = 0.0, 0.0
        w1, h1 = random.uniform(self.min_length, 1.0 - self.min_length), random.uniform(self.min_length, 1.0 - self.min_length)
        w2, h2 = 1.0 - (x + w1), 1.0 - (y + h1)
        l_shape = [
            PartLine((x, y), (x, y + h1), (-1, 0)),  # Left
            PartLine((x, y + h1), (x + w1, y + h1), (0, 1)),  # Bottom
            PartLine((x + w1, y + h1), (x + w1, y + h1 + h2), (-1, 0)),  # Left
            PartLine((x + w1, y + h1 + h2), (x + w1 + w2, y + h1 + h2), (0, 1)),  # Bottom
            PartLine((x + w1 + w2, y + h1 + h2), (x + w1 + w2, y), (1, 0)),  # Right
            PartLine((x + w1 + w2, y), (x, y), (0, -1))  # Top
        ]
        permutation_indices = list(range(len(l_shape)))  # shuffle l-shape here as id elements won't be shuffled later.
        random.shuffle(permutation_indices)
        ids = [str(i) for i in range(6)]
        shuffled_l_shape = [l_shape[i] for i in permutation_indices]    # shuffle l-shapes to have permutation
        # shuffle id in other direction, think about it!
        permutation_indices_inv = {i: j for i, j in zip(permutation_indices, range(6))}
        shuffled_ids = [ ids[permutation_indices_inv[i]] for i in range(6)]
        l_shape_with_id = [PartLineWithId(id, line) for id, line in zip(ids, shuffled_l_shape)]
        # connect each line with its neighbors using ConnectTwoElementsWithId
        connections = []
        for i in range(6):
            connections.append(ConnectTwoElementsWithId(shuffled_ids[i], shuffled_ids[(i + 1) % 6]))
        return l_shape_with_id + connections

    def action_types(self) -> Set[Type[Element]]:
        return {PartLineWithId, ConnectTwoElementsWithId}

    @property
    def n_actions(self) -> int:
        return 6 * 2


class RectilinearOutlineGenerator(Generator):
    """
    A generator that generates the outline of a general rectilinear part.

    The part is grown as a polyomino on a `grid` x `grid` lattice and its boundary is traced, so it
    is a single simple rectilinear polygon by construction: one connected area, no holes, no
    self-intersection and no vertex where the boundary touches itself. Collinear boundary runs are
    merged, so no two consecutive edges lie on the same line and the decomposition of a stroke into
    edges is never ambiguous. Two edges that do not share a vertex are at least one cell apart,
    which is what keeps two strokes from reading as one.

    `n_cells` is the difficulty: more cells make a longer, more involved boundary. The l-shape is
    the special case of a boundary with six edges, so this generator is the same task one rung up.

    Every constraint is checked rather than assumed, and a candidate that fails any of them is
    thrown away and drawn again — an outline that reached the dataset unchecked would be a record
    the image does not determine.
    """

    def __init__(
        self,
        grid: int = 8,
        n_cells: int = 12,
        min_edges: int = 8,
        max_edges: int = 24,
        min_edge_cells: int = 1,
        max_attempts: int = 200,
    ):
        assert grid >= 3, "grid must leave room for a part"
        assert 1 <= n_cells <= grid * grid, "n_cells must fit on the grid"
        assert 4 <= min_edges <= max_edges, "an outline has at least four edges"
        assert min_edge_cells >= 1, "an edge spans at least one cell"
        self.grid = grid
        self.n_cells = n_cells
        self.min_edges = min_edges
        self.max_edges = max_edges
        self.min_edge_cells = min_edge_cells
        self.max_attempts = max_attempts

    # -- growing the area ------------------------------------------------------------------

    def _grow(self) -> Set[Tuple[int, int]]:
        """A connected set of `n_cells` cells, grown one orthogonal neighbour at a time."""
        start = (random.randrange(self.grid), random.randrange(self.grid))
        cells = {start}
        frontier = self._neighbours(start)
        while len(cells) < self.n_cells and frontier:
            cell = random.choice(sorted(frontier))
            frontier.discard(cell)
            if cell in cells:
                continue
            cells.add(cell)
            frontier |= {n for n in self._neighbours(cell) if n not in cells}
        return cells

    def _neighbours(self, cell: Tuple[int, int]) -> Set[Tuple[int, int]]:
        x, y = cell
        return {
            (nx, ny)
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
            if 0 <= nx < self.grid and 0 <= ny < self.grid
        }

    def _has_hole(self, cells: Set[Tuple[int, int]]) -> bool:
        """Whether the complement has a pocket the outside cannot reach, which is a hole."""
        seen = set()
        stack = [
            (x, y)
            for x in range(-1, self.grid + 1)
            for y in (-1, self.grid)
            if (x, y) not in cells
        ] + [
            (x, y)
            for y in range(-1, self.grid + 1)
            for x in (-1, self.grid)
            if (x, y) not in cells
        ]
        while stack:
            cell = stack.pop()
            if cell in seen or cell in cells:
                continue
            x, y = cell
            if not (-1 <= x <= self.grid and -1 <= y <= self.grid):
                continue
            seen.add(cell)
            stack.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))
        outside = (self.grid + 2) * (self.grid + 2) - len(cells)
        return len(seen) != outside

    def _has_pinch(self, cells: Set[Tuple[int, int]]) -> bool:
        """Two cells meeting only at a corner, where the boundary would touch itself at a point."""
        for x, y in cells:
            for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                if (x + dx, y + dy) in cells and (x + dx, y) not in cells and (x, y + dy) not in cells:
                    return True
        return False

    # -- tracing its boundary --------------------------------------------------------------

    def _boundary(self, cells: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """The boundary as a cycle of lattice points, collinear runs already merged."""
        # every cell edge with the area on one side and nothing on the other, directed so the area
        # stays on the left and the cycle therefore closes
        edges: dict = {}
        for x, y in cells:
            if (x, y - 1) not in cells:
                edges[(x, y)] = (x + 1, y)
            if (x + 1, y) not in cells:
                edges[(x + 1, y)] = (x + 1, y + 1)
            if (x, y + 1) not in cells:
                edges[(x + 1, y + 1)] = (x, y + 1)
            if (x - 1, y) not in cells:
                edges[(x, y + 1)] = (x, y)
        if not edges:
            return []
        start = min(edges)
        cycle = [start]
        current = edges[start]
        while current != start:
            if current not in edges or len(cycle) > 4 * len(cells) + 4:
                return []  # not a single closed boundary
            cycle.append(current)
            current = edges[current]
        if len(cycle) != len(edges):
            return []  # the boundary fell into more than one loop
        return self._merge_collinear(cycle)

    @staticmethod
    def _merge_collinear(cycle: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Drops every point whose neighbours lie on the same line, so consecutive edges turn."""
        merged = []
        n = len(cycle)
        for i in range(n):
            before, here, after = cycle[i - 1], cycle[i], cycle[(i + 1) % n]
            turns = (here[0] - before[0], here[1] - before[1]) != (after[0] - here[0], after[1] - here[1])
            if turns:
                merged.append(here)
        return merged

    # -- the outline it stands for ---------------------------------------------------------

    def _outline(self) -> Optional[List[PartLine]]:
        cells = self._grow()
        if len(cells) < self.n_cells or self._has_hole(cells) or self._has_pinch(cells):
            return None
        corners = self._boundary(cells)
        if not (self.min_edges <= len(corners) <= self.max_edges):
            return None

        # normalise the lattice to the unit square, keeping the part square rather than stretched
        xs = [x for x, _ in corners]
        ys = [y for _, y in corners]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        if span <= 0:
            return None
        cell = 1.0 / span
        if cell * self.min_edge_cells <= 0.0:
            return None
        origin = (min(xs), min(ys))

        def point(p: Tuple[int, int]) -> Tuple[float, float]:
            return ((p[0] - origin[0]) * cell, (p[1] - origin[1]) * cell)

        lines = []
        n = len(corners)
        for i in range(n):
            a, b = corners[i], corners[(i + 1) % n]
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) < self.min_edge_cells:
                return None  # an edge shorter than the smallest we allow
            lines.append(PartLine(point(a), point(b), self._outward(a, b)))
        return lines

    @staticmethod
    def _outward(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[float, float]:
        """The normal pointing away from the area, for the edge running from `a` to `b`.

        The boundary is traced with the area on the left, so the outward side is to the right.
        """
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = abs(dx) + abs(dy)
        return (dy / length, -dx / length)

    # -- the record ------------------------------------------------------------------------

    def get_actions(self) -> List[Element]:
        outline = None
        for _ in range(self.max_attempts):
            outline = self._outline()
            if outline is not None:
                break
        if outline is None:
            raise RuntimeError(
                f"no outline of {self.min_edges}..{self.max_edges} edges from {self.n_cells} cells "
                f"on a {self.grid}x{self.grid} grid in {self.max_attempts} attempts"
            )

        # the same shuffle the l-shape does, so a record carries no order of its own
        n = len(outline)
        permutation_indices = list(range(n))
        random.shuffle(permutation_indices)
        ids = [str(i) for i in range(n)]
        shuffled = [outline[i] for i in permutation_indices]
        permutation_indices_inv = {i: j for i, j in zip(permutation_indices, range(n))}
        shuffled_ids = [ids[permutation_indices_inv[i]] for i in range(n)]
        with_id = [PartLineWithId(id, line) for id, line in zip(ids, shuffled)]
        connections = [
            ConnectTwoElementsWithId(shuffled_ids[i], shuffled_ids[(i + 1) % n]) for i in range(n)
        ]
        return with_id + connections

    def action_types(self) -> Set[Type[Element]]:
        return {PartLineWithId, ConnectTwoElementsWithId}

    @property
    def n_actions(self) -> int:
        return self.max_edges * 2


class AddIdGenerator(GeneratorDecorator):

    def __init__(self, generator: Generator):
        super().__init__(generator)

    def get_actions(self) -> List[Element]:
        result = []
        for id, action in enumerate(self.generator.get_actions()):
            if isinstance(action, PartLine):
                result.append(PartLineWithId(
                    str(id),
                    action,
                ))
            else:
                result.append(action)
        return result

    def action_types(self) -> Set[Type[Element]]:
        return self.generator.action_types() - {PartLine} | {PartLineWithId}

    @property
    def n_actions(self) -> int:
        return self.generator.n_actions


class AnnotateLineGenerator(GeneratorDecorator):
    def __init__(self, generator: Generator, ratio: float = 0.5, max_line_length: float = 0.1, min_part_line_length: float = 0.2, text_offset=0.05, graph_mode=True):
        super().__init__(generator)
        self.ratio = ratio
        self.line_length_dist = TruncatedNormalSampler(loc=max_line_length * (3 / 4), scale=0.1, min_value=max_line_length / 2, max_value=max_line_length)
        self.min_part_line_length = min_part_line_length
        self.text_offset_dist = TruncatedNormalSampler(loc=text_offset, scale=0.1, min_value=text_offset / 2, max_value=text_offset * 2)
        self.graph_mode = graph_mode

    def get_actions(self) -> List[Element]:
        actions = self.generator.get_actions()
        annotations = []
        for action in actions:
            if self.graph_mode and isinstance(action, PartLine):
                continue  # Do not annotate PartLine in graph mode
            action_id = None
            if self.graph_mode and isinstance(action, PartLineWithId):
                action_id = action.id
                action = action.part_line
            if isinstance(action, PartLine) and action.out_dir is not None and action.length > self.min_part_line_length:
                if random.random() < self.ratio:
                    line_length = self.line_length_dist()
                    text_offset = line_length / 2
                    start_point_offset = np.array(action.start_point) + np.array(action.out_dir) * line_length
                    end_point_offset = np.array(action.end_point) + np.array(action.out_dir) * line_length
                    start_point_offset = _to_point(start_point_offset)
                    end_point_offset = _to_point(end_point_offset)
                    annotations.append(HelpLine(action.start_point, start_point_offset, formative_only=action_id is not None))
                    annotations.append(HelpLine(action.end_point, end_point_offset, formative_only=action_id is not None))

                    annotations.append(BothSidedArrow(start_point_offset, end_point_offset, formative_only=action_id is not None))

                    mid_point = (
                        (start_point_offset[0] + end_point_offset[0]) / 2,
                        (start_point_offset[1] + end_point_offset[1]) / 2,
                    )
                    #bottom
                    if np.allclose(np.array(action.out_dir), np.array([0, 1])):
                        text = AnnotationText((mid_point[0], mid_point[1] - text_offset), text_angle=0)
                    #top
                    elif np.allclose(np.array(action.out_dir), np.array([0, -1])):
                        text = AnnotationText((mid_point[0], mid_point[1] - text_offset), text_angle=0)
                    #left
                    elif np.allclose(np.array(action.out_dir), np.array([-1, 0])):
                        text = AnnotationText((mid_point[0] - text_offset, mid_point[1]), text_angle=90)
                    #right
                    else:
                        assert np.allclose(np.array(action.out_dir), np.array([1, 0]))
                        text = AnnotationText((mid_point[0] - text_offset, mid_point[1]), text_angle=90)

                    if action_id is not None:
                        text = AnnotationTextRefId(action_id, text)

                    annotations.append(text)
        return actions + annotations

    def action_types(self) -> Set[Type[Element]]:
        if self.graph_mode:
            return {AnnotationTextRefId} | self.generator.action_types()
        else:
            return {HelpLine, BothSidedArrow, AnnotationText} | self.generator.action_types()

    @property
    def n_actions(self) -> int:
        if self.graph_mode:
            return self.generator.n_actions + self.generator.n_actions
        else:
            return self.generator.n_actions + self.generator.n_actions * 4


class AnnotateRotatedLineGenerator(Generator):
    def __init__(self, generator: Generator, ratio: float = 0.5, line_length: float = 0.1, arrow_type: str = "ArrowedLine"):
        self.generator = generator
        self.ratio = ratio
        self.line_length = line_length
        self.arrow_type = arrow_type

    def get_actions(self) -> List[Element]:
        actions = self.generator.get_actions()
        annotations = []

        for action in actions:
            if isinstance(action, PartLine) and action.out_dir is not None:
                if random.random() < self.ratio:
                    start_point_offset = np.array(action.start_point) + np.array(action.out_dir) * self.line_length
                    end_point_offset = np.array(action.end_point) + np.array(action.out_dir) * self.line_length
                    start_point_offset = _to_point(start_point_offset)
                    end_point_offset = _to_point(end_point_offset)
                    annotations.append(HelpLine(action.start_point, start_point_offset))
                    annotations.append(HelpLine(action.end_point, end_point_offset))

                    annotations.append(BothSidedArrow(start_point_offset, end_point_offset))
                    mid_point = (
                        (start_point_offset[0] + end_point_offset[0]) / 2,
                        (start_point_offset[1] + end_point_offset[1]) / 2,
                    )

                    up_vector = np.array([0, -1])
                    text_angle = np.degrees(
                        np.arccos(np.dot(up_vector, action.out_dir / np.linalg.norm(action.out_dir)))
                    )

                    text_point = (
                        mid_point[0] + action.out_dir[0] * (self.line_length / 2),
                        mid_point[1] + action.out_dir[1] * (self.line_length / 2),
                    )
                    annotations.append(AnnotationText(text_point, text_angle=text_angle * np.sign(action.out_dir[0])))
        return actions + annotations

    def action_types(self) -> Set[Type[Element]]:
        return {HelpLine, BothSidedArrow, AnnotationText} | self.generator.action_types()

    @property
    def n_actions(self) -> int:
        return self.generator.n_actions * 5


class MirrorAugmentationGenerator(GeneratorDecorator):
    """
    A generator that augments the given generator by mirroring the generated actions.
    """

    def __init__(self, generator: Generator, mirror_x: bool = False, mirror_y: bool = False):
        super().__init__(generator)
        self.mirror_x = mirror_x
        self.mirror_y = mirror_y

    def get_actions(self) -> List[Element]:
        original_actions = self.generator.get_actions()
        mirrored_actions = []

        for original_action in original_actions:
            action = original_action
            if self.mirror_x:
                action = original_action.mirror_x()
            if self.mirror_y:
                action = original_action.mirror_y()
            assert action is not None, f"Action {action} is None for {self.mirror_x} {self.mirror_y} and {original_action}"
            mirrored_actions.append(action)

        return mirrored_actions


class RotationGenerator(GeneratorDecorator):
    """
    A generator that augments the given generator by rotating the generated actions.
    """

    def __init__(self, generator: Generator, angle: float, center: Tuple[float, float] = (0.5, 0.5)):
        super().__init__(generator)
        self.angle = angle
        self.center = center

    def get_actions(self) -> List[Element]:
        original_actions = self.generator.get_actions()
        rotated_actions = [action.rotate(self.angle, self.center) for action in original_actions]
        return rotated_actions


class RandomRotationAugmentation(GeneratorDecorator):

    """
    A generator that augments the given generator by rotating the generated actions randomly by 0, 90, 180 or 270 degrees.
    """

    def __init__(self, generator):
        super().__init__(generator)
        self.rotations = [0, 90, 180, 270]

    def get_actions(self) -> List[Element]:
        angle = random.choice(self.rotations)
        return RotationGenerator(self.generator, angle).get_actions()


class RandomMirrorAugmentation(GeneratorDecorator):

    """
    A generator that augments the given generator by randomly mirroring the generated actions.
    """

    def __init__(self, generator):
        super().__init__(generator)

    def get_actions(self) -> List[Element]:
        mirror_x = random.choice([True, False])
        mirror_y = random.choice([True, False])
        return MirrorAugmentationGenerator(self.generator, mirror_x, mirror_y).get_actions()


class PickOneGenerator(Generator):
    """
    A generator that randomly picks one of the given generators and returns its actions.
    """

    def __init__(self, generators: List[Generator]):
        self.generators = generators

    def get_actions(self):
        return random.choice(self.generators).get_actions()

    def action_types(self) -> Set[Type[Element]]:
        return set.union(*(generator.action_types() for generator in self.generators))

    @property
    def n_actions(self) -> int:
        return max(generator.n_actions for generator in self.generators)


class PickAllGenerator(Generator):
    """
    A generator that picks all the given generators and returns all actions concatenated.
    """

    def __init__(self, generators: List[Generator]):
        self.generators = generators

    def get_actions(self):
        actions = []
        for generator in self.generators:
            actions.extend(generator.get_actions())
        return actions

    def action_types(self) -> Set[Type[Element]]:
        return set.union(*(generator.action_types() for generator in self.generators))

    @property
    def n_actions(self) -> int:
        return sum(generator.n_actions for generator in self.generators)


class MultiGenerator(GeneratorDecorator):

    """
    A generator that combines the actions of all given generators.
    """

    def __init__(self, generator: Generator, num_generators: int = 2):
        super().__init__(generator)
        self.num_generators = num_generators

    def get_actions(self):
        results = []
        for _ in range(self.num_generators):
            results.extend(self.generator.get_actions())
        return results

    @property
    def n_actions(self) -> int:
        return self.num_generators * self.generator.n_actions


class MarginGenerator(GeneratorDecorator):
    """
    Generates a margin around the actions of the given generator.
    """

    def __init__(self, generator: Generator, margin: float = 0.1):
        super().__init__(generator)
        self.margin = margin

    def get_actions(self) -> List[Element]:
        actions = self.generator.get_actions()
        return [action.scale(1 - 2 * self.margin).move(self.margin, self.margin) for action in actions]


class RandomMarginGenerator(GeneratorDecorator):
    """
    Generates a margin around the actions of the given generator.
    """

    def __init__(self, generator: Generator, min_margin: float = 0.1, max_margin: float = 0.25):
        super().__init__(generator)
        self.min_margin = min_margin
        self.max_margin = max_margin

    def get_actions(self) -> List[Element]:
        margin = random.uniform(self.min_margin, self.max_margin)
        actions = self.generator.get_actions()
        return [action.scale(1 - 2 * margin).move(margin, margin) for action in actions]


class JitterGenerator(GeneratorDecorator):
    """
    A generator that jitters/moves the generated actions by a small amount.
    """

    def __init__(self, generator: Generator, jitter: float = 0.05):
        super().__init__(generator)
        self.jitter = jitter

    def get_actions(self) -> List[Element]:
        actions = self.generator.get_actions()
        return [action.move(random.uniform(-self.jitter, self.jitter), random.uniform(-self.jitter, self.jitter)) for action in actions]


class GridGenerator(GeneratorDecorator):

    """
    A generator that generates actions of the given generator multiple times in a grid.

    For example, it can draw 4 L-Shapes in a 2x2 grid using GridGenerator(LShapeGenerator(), 2, 2).
    """

    def __init__(self, generator: Generator, num_rows: int = 2, num_columns: int = 2):
        self.generator = generator
        self.num_rows = num_rows
        self.num_columns = num_columns

    def get_actions(self) -> List[Element]:
        coordinates_per_cell = [
            ((i / self.num_columns, j / self.num_rows), ((i + 1) / self.num_columns, (j + 1) / self.num_rows))
            for i in range(self.num_columns) for j in range(self.num_rows)
        ]
        actions_per_grid = [self.generator.get_actions() for _ in range(len(coordinates_per_cell))]
        result = []
        for actions, coordinates in zip(actions_per_grid, coordinates_per_cell):
            min_x, min_y = coordinates[0]
            max_x, max_y = coordinates[1]
            result.extend(
                action.move_and_scale_to(max_x, max_y, min_x, min_y) for action in actions
            )
        return result

    @property
    def n_actions(self) -> int:
        return self.num_rows * self.num_columns * self.generator.n_actions


class MinLengthGenerator(GeneratorDecorator):

    """
    A generator that keeps generating actions until a minimum length requirement is satisfied by each action.
    """

    def __init__(self, generator: Generator, min_length: float = 0.1):
        super().__init__(generator)
        self.min_length = min_length

    def get_actions(self) -> List[Element]:
        def min_line_length(actions: List[Element]) -> float:
            min_length = 1000
            for action in actions:
                if isinstance(action, PartLine):
                    length = np.linalg.norm(np.array(action.start_point) - np.array(action.end_point))
                    min_length = min(min_length, length)
            return min_length
        actions = self.generator.get_actions()
        while min_line_length(actions) < self.min_length:
            actions = self.generator.get_actions()
        return actions


class ExactNumberOfActions(GeneratorDecorator):

    """
    A generator that keeps generating actions until the exact number of actions is satisfied.
    """

    def __init__(self, generator: Generator, num_actions: int = 4):
        super().__init__(generator)
        self.num_actions = num_actions

    def get_actions(self) -> List[Element]:
        actions = self.generator.get_actions()
        while len(actions) != self.num_actions:
            actions = self.generator.get_actions()
        return actions

    @property
    def n_actions(self) -> int:
        return self.num_actions


class RandomTranslationGenerator(GeneratorDecorator):

    def __init__(self, generator: Generator, max_translation: float = 0.1, min_translation: float = 0.0, exclude: Tuple[float, float] = None):
        super().__init__(generator)
        self.max_translation = max_translation
        self.min_translation = min_translation
        self.exclude = exclude

    def get_actions(self) -> List[Element]:
        def get_random_value():
            value = random.uniform(0, self.max_translation - self.min_translation) + self.min_translation
            while self.exclude is not None and self.exclude[0] <= value <= self.exclude[1]:
                value = random.uniform(0, self.max_translation - self.min_translation) + self.min_translation
            return np.random.choice([value, -value])
        dx, dy = get_random_value(), get_random_value()
        actions = self.generator.get_actions()
        return [action.move(dx, dy) for action in actions]


def horizontal_generator(n: int, generator: Generator) -> Generator:
    return GridGenerator(
        MarginGenerator(
            generator,
            margin=0.25
        ), num_rows=n, num_columns=1,
    )


def vertical_generator(n: int, generator: Generator) -> Generator:
    return GridGenerator(
        MarginGenerator(
            generator,
            # MinLengthGenerator(ConvexHullPolygonGenerator2(12, min_inner_angle=170), min_length=0.1),
            margin=0.25
        ), num_rows=1, num_columns=n,
    )


def vertical_line_generator(n: int, **kwargs) -> Generator:
    return vertical_generator(n, RotationGenerator(HorizontalLineGenerator(**kwargs), angle=90))


def horizontal_line_generator(n: int, **kwargs) -> Generator:
    return horizontal_generator(n, HorizontalLineGenerator(**kwargs))


def random_mirror_and_rotation_augmentation(generator: Generator) -> Generator:
    return RandomMirrorAugmentation(RandomRotationAugmentation(
        generator
    ))


class AppendFinishDrawingGenerator(GeneratorDecorator):

    def __init__(self, generator: Generator):
        super().__init__(generator)
        self.finish_drawing = FinishDrawing()

    def get_actions(self) -> List[Element]:
        actions = self.generator.get_actions()
        actions.append(self.finish_drawing)
        return actions

    def action_types(self) -> Set[Type[Element]]:
        return {FinishDrawing} | self.generator.action_types()

    @property
    def n_actions(self) -> int:
        return 1 + self.generator.n_actions


class ShuffleByElementGenerator(GeneratorDecorator):

    def __init__(self, generator):
        super().__init__(generator)
        self.type_order = [PartLine, PartLineWithId, Circle, HelpLine, BothSidedArrow, ConnectTwoElementsWithId, AnnotationText, AnnotationTextRefId]
        self.type_with_id = [PartLineWithId]

    def get_actions(self) -> List[Element]:
        actions = self.generator.get_actions()
        actions_by_type = defaultdict(list)
        for action in actions:
            actions_by_type[action.__class__.__name__].append(action)
        shuffled_actions = []
        for element_class in self.type_order:
            t = actions_by_type[element_class.__name__]
            if element_class not in self.type_with_id:
                # shuffle elements without id
                random.shuffle(t)
            shuffled_actions.extend(t)
        missing_order = set(actions_by_type.keys()) - set(c.__name__ for c in self.type_order)
        if len(missing_order) > 0:
            print(f"WARNING: Missing order: {missing_order} - elements will be dropped")
        return shuffled_actions


class SortByCoordinateByElementGenerator(GeneratorDecorator):

    def __init__(self, generator):
        super().__init__(generator)
        self.type_order = [PartLine, Circle, HelpLine, BothSidedArrow, AnnotationText]

    def get_actions(self) -> List[Element]:
        actions = self.generator.get_actions()
        actions_by_type = defaultdict(list)
        for action in actions:
            actions_by_type[action.__class__.__name__].append(action)
        sorted_actions = []
        for element_class in self.type_order:
            t = actions_by_type[element_class.__name__]
            t.sort(key=lambda x: x._coordinates_params[0][0])
            sorted_actions.extend(t)
        return sorted_actions


class ShuffleGenerator(GeneratorDecorator):

    def get_actions(self) -> List[Element]:
        actions = self.generator.get_actions()
        random.shuffle(actions)
        return actions