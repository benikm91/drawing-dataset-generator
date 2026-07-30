import abc
import os
import random
from collections import defaultdict
from functools import partial

import cv2
import numpy as np
from typing import Any, List, Optional
from PIL import ImageFont

from generator import Element, PartLine, Circle, PartLineWithId, BothSidedArrow
from random_f import TruncatedNormalSampler, UniformSampler


def _font_size(font: Any) -> int:
    return int(getattr(font, "size", 8))


def _thickness_for(thickness_by_type: Any, action: Element) -> int:
    return int(thickness_by_type.get(action.__class__, 1))


def _sample_thickness(thickness_dists: Any, action: Element) -> int:
    return int(thickness_dists[action.__class__]())


def _load_font(path: str, font_size: int):
    if not os.path.exists(path):
        return None
    return ImageFont.truetype(path, font_size)


def _fallback_font(font_size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def _build_font_pool(font_sizes: List[int]) -> List[Any]:
    font_paths = [
        os.path.join(os.path.dirname(__file__), '../res/unifont-15.0.01.otf'),
        os.path.join(os.path.dirname(__file__), '../res/arial/ARIAL.TTF'),
        os.path.join(os.path.dirname(__file__), '../res/times-new-roman.ttf'),
        os.path.join(os.path.dirname(__file__), '../res/calibri.ttf'),
    ]
    fonts = [
        font
        for font_size in font_sizes
        for font in (_load_font(path, font_size) for path in font_paths)
        if font is not None
    ]
    return fonts or [_fallback_font(font_size) for font_size in font_sizes]

class Renderer(abc.ABC):

    def __init__(self, canvas_width: int, canvas_height: int):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def draw(self, actions: List[Element], seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        return self._draw(actions)

    @abc.abstractmethod
    def _draw(self, actions: List[Element]) -> np.ndarray:
        pass


def one_f():
    return 1


class StaticRenderer(Renderer):

    def __init__(self, canvas_width: int, canvas_height: int, thickness=None, font_size=8):
        super(StaticRenderer, self).__init__(canvas_width, canvas_height)
        if thickness is not None:
            self.thickness = thickness
        else:
            self.thickness = defaultdict(one_f, {
                PartLine: 1,
                Circle: 1,
            })
        FONT_PATH = os.path.join(os.path.dirname(__file__), '../res/unifont-15.0.01.otf')
        if os.path.exists(FONT_PATH):
            self.font = ImageFont.truetype(FONT_PATH, font_size)
        else:
            self.font = _fallback_font(font_size)

    def _draw(self, actions: List[Element]) -> np.ndarray:
        img = np.zeros((self.canvas_width, self.canvas_height), dtype=np.float32)
        for action in actions:
            action.draw(
                img,
                thickness=self.thickness[action.__class__],
                color=255,
                arrow_type="FilledTriangleArrow",
                font=self.font,
                font_size=_font_size(self.font),
            )
        return img


class StaticGraphRenderer(StaticRenderer):

    def _draw(self, actions: List[Element]) -> np.ndarray:
        def action_center_to_img(action):
            if action is None:
                return None
            if action.center == (-1, -1):
                return None
            return int(round(action.center[0] * img.shape[1])), int(round(action.center[1] * img.shape[0]))
        img = np.zeros((self.canvas_width, self.canvas_height), dtype=np.float32)
        relation_img = np.zeros_like(img)
        action_by_id = {getattr(action, 'id'): action for action in actions if hasattr(action, 'id')}
        for i, action in enumerate(actions):
            action.draw(img, thickness=1, color=255, arrow_type="FilledTriangleArrow", font=self.font, font_size=_font_size(self.font))
            refs = list(action.refs)
            if len(refs) > 0:
                if len(refs) == 1:
                    # action references another action, draw from action to referenced action
                    from_c = action_center_to_img(action)
                else:
                    # action connects two actions, draw from first to second action
                    assert len(refs) == 2
                    from_c = action_center_to_img(action_by_id.get(refs.pop(), None))
                to_c = action_center_to_img(action_by_id.get(refs.pop(), None))
                if from_c is not None and to_c is not None:
                    BothSidedArrow._draw_filled_triangle(
                        relation_img, from_c, to_c, color=255, thickness=1
                    )
        return np.dstack([img, relation_img, np.zeros_like(img)])


class InOrderStaticRenderer(StaticRenderer):

    def _draw(self, actions: List[Element]) -> np.ndarray:
        def interpolate(f: int, t: int, i: int, n: int) -> int:
            return int(f + (t - f) * i / n)
        img = np.zeros((self.canvas_width, self.canvas_height), dtype=np.float32)
        for i, action in enumerate(actions):
            action.draw(img, thickness=_thickness_for(self.thickness, action), color=interpolate(32, 255, i, len(actions)), arrow_type="FilledTriangleArrow", font=self.font, font_size=8)
        return img


def default_sampler():
    return TruncatedNormalSampler(1, scale=0.5, min_value=1, integer=True)


def post_nothing(img: np.ndarray) -> np.ndarray:
    return img


class NoisyRenderer(Renderer):

    def __init__(
        self,
        canvas_width: int,
        canvas_height: int,
        thickness_loc: float = 2,
        thickness_scale: float = 0.5,
        min_thickness: int = 1,
        arrow_mode: str = "random",
        arrow_type: str = "FilledTriangleArrow",
        color_loc: float = 255,
        color_scale: float = 15,
        color_min: int = 128,
        color_max: int = 255,
        blur_mode: str = "random",
        font_mode: str = "random",
        font_size: int = 14,
    ):
        super(NoisyRenderer, self).__init__(canvas_width, canvas_height)
        font_sizes = [10, 12, 14, 16, 18, 20]
        self.fonts = _build_font_pool(font_sizes)
        self.fixed_font = _build_font_pool([font_size])[0]

        self.thickness_dists = defaultdict(default_sampler, {
            PartLine: TruncatedNormalSampler(loc=thickness_loc, scale=thickness_scale, min_value=min_thickness, integer=True),
            Circle: TruncatedNormalSampler(loc=thickness_loc, scale=thickness_scale, min_value=min_thickness, integer=True),
        })
        self.arrow_mode = arrow_mode
        self.arrow_type = arrow_type
        self.arrow_type_dist = UniformSampler(["ArrowedLine", "FilledTriangleArrow", "NonFilledTriangleArrow"])
        self.color_dist = TruncatedNormalSampler(loc=color_loc, scale=color_scale, min_value=color_min, max_value=color_max, integer=True)
        self.font_mode = font_mode
        self.font_dist = UniformSampler(self.fonts)

        self.blur_mode = blur_mode
        self.blur_options = {
            "none": post_nothing,
            "3x3": partial(cv2.GaussianBlur, ksize=(3, 3), sigmaX=0),
            "5x5": partial(cv2.GaussianBlur, ksize=(5, 5), sigmaX=0),
        }

    def _draw(self, actions: List[Element]) -> np.ndarray:
        img = np.zeros((self.canvas_width, self.canvas_height), dtype=np.float32)
        thicknesses = {action.__class__: _sample_thickness(self.thickness_dists, action) for action in actions}
        arrow_type = self.arrow_type_dist() if self.arrow_mode == "random" else self.arrow_type
        color = self.color_dist()
        font = self.font_dist() if self.font_mode == "random" else self.fixed_font
        for action in actions:
            action.draw(
                img,
                thickness=int(thicknesses.get(action.__class__, 1)),
                color=color,
                arrow_type=arrow_type,
                font=font,
                font_size=_font_size(font),
            )

        if self.blur_mode == "random":
            img = random.choice([post_nothing, post_nothing, self.blur_options["3x3"], self.blur_options["5x5"]])(img)
        else:
            img = self.blur_options.get(self.blur_mode, post_nothing)(img)
        img = np.clip(img, 0, 255)

        return img