from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import random
import matplotlib.pyplot as plt
from shapely.ops import unary_union, split, substring
from shapely.geometry import Polygon, MultiPoint, GeometryCollection,  LineString, Point, MultiLineString
from shapely.affinity import translate
from matplotlib.patches import Circle, Polygon as MplPolygon


Pose = Tuple[float, float, float]  # (x, y, yaw)

XY = Tuple[float, float]  # (x, y)

@dataclass
class Region:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def to_list(self) -> List[float]:
        return [self.x_min, self.x_max, self.y_min, self.y_max]

    def draw(self,
             ax: Optional[plt.Axes] = None,
             row: Optional[int] = None,
             col: Optional[int] = None,
             idx: Optional[int] = None,
             **kwargs) -> plt.Axes:
        
        if ax is None:
            fig, ax = plt.subplots()
        # 绘制矩形
        rect = plt.Rectangle((self.x_min, self.y_min),
                             self.x_max - self.x_min,
                             self.y_max - self.y_min,
                             fill=False,
                             **kwargs)
        ax.add_patch(rect)

        # 如果传入了 row/col/idx，就在右下角标注
        if row is not None and col is not None and idx is not None:
            # 右下角坐标略微内缩一点
            tx = self.x_max - 0.1 * (self.x_max - self.x_min)
            ty = self.y_min + 0.05 * (self.y_max - self.y_min)
            ax.text(tx, ty,
                    f"r{row},c{col}\n#{idx}",
                    ha='right', va='bottom',
                    fontsize=8, color=kwargs.get('edgecolor','black'))

        ax.set_aspect('equal', 'box')
        # ax.set_xlim(self.x_min - 1, self.x_max + 1)
        # ax.set_ylim(self.y_min - 1, self.y_max + 1)
        return ax


@dataclass
class Cylinder:
    """Represents a cylindrical obstacle's center position and radius."""
    x: float
    y: float
    radius: float = 0.02

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.radius]

    def draw(self,
             ax: Optional[plt.Axes] = None,
             edgecolor: str = 'black',
             facecolor: str = 'none',
             linewidth: float = 1.0,
             **kwargs) -> plt.Axes:
        """
        Draw this cylinder as a circle in the XY plane.
        """
        if ax is None:
            fig, ax = plt.subplots()
        circ = Circle((self.x, self.y),
                      radius=self.radius,
                      edgecolor=edgecolor,
                      facecolor=facecolor,
                      linewidth=linewidth,
                      **kwargs)
        ax.add_patch(circ)
        ax.set_aspect('equal', 'box')
        return ax

@dataclass
class Wall:
    """
    对应 SDF 中一个 <box> 的 size + pose：
      length    — 沿局部 X 轴的尺寸
      thickness — 沿局部 Y 轴的尺寸（即墙体厚度）
      height    — 沿 Z 轴的尺寸
      pose      — (x, y, yaw) 仅用于在平面上定位
    """
    length: float
    thickness: float
    height: float
    pose: Tuple[float, float, float]  # (x, y, yaw)

    def draw(self,
             ax: Optional[plt.Axes] = None,
             edgecolor: str = 'black',
             facecolor: str = 'none',
             linewidth: float = 1.0,
             **kwargs) -> plt.Axes:
        """
        Draw this wall as a rotated rectangle in the XY plane,
        ignoring the height dimension.
        """
        if ax is None:
            fig, ax = plt.subplots()

        x, y, yaw = self.pose
        L, T = self.length, self.thickness
        # 定义局部四个角 (以中心为原点)
        half_L, half_T = L/2, T/2
        local_corners = [
            (+half_L, +half_T),
            (+half_L, -half_T),
            (-half_L, -half_T),
            (-half_L, +half_T),
        ]
        # 旋转并平移到全局
        c, s = math.cos(yaw), math.sin(yaw)
        global_corners = [
            (x + lx*c - ly*s, y + lx*s + ly*c)
            for lx, ly in local_corners
        ]
        poly = MplPolygon(global_corners,
                       closed=True,
                       edgecolor=edgecolor,
                       facecolor=facecolor,
                       linewidth=linewidth,
                       **kwargs)
        ax.add_patch(poly)
        ax.set_aspect('equal', 'box')
        return ax

    




@dataclass
class Configuration:
    """Full specification for one training scenario."""
    density: str                  # 'continuous' or 'sparse'
    region: Region
    walls:    List[Wall]          # 如果 density=='continuous' 时，这里存 Wall 列表
    cylinders: List[Cylinder]     # 如果 density=='sparse'     时，这里存 Cylinder 列表
    start_pose: Pose
    target_position:   XY

    def to_dict(self) -> dict:
        return {
            'density':    self.density,
            'region':     self.region.to_list(),
            'walls':      [ [w.length, w.thickness, w.height] for w in self.walls ],
            'cylinders':  [c.to_list() for c in self.cylinders],
            'start_pose': list(self.start_pose),
            'target_position':   list(self.target_position),
        }