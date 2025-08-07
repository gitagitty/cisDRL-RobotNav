from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import random
import matplotlib.pyplot as plt
from shapely.ops import unary_union, split, substring
from shapely.geometry import Polygon, MultiPoint, GeometryCollection,  LineString, Point, MultiLineString
from shapely.affinity import translate
from matplotlib.patches import Circle, Polygon as MplPolygon

from config_types import Pose, XY, Region, Cylinder, Wall, Configuration

import json







@dataclass
class Envelope:
    """
    Buffered convex hull envelope of swept area.
    """
    polygon: Polygon

    def draw(self, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        plot_kwargs = kwargs.copy()
        if 'edgecolor' in plot_kwargs:
            plot_kwargs['color'] = plot_kwargs.pop('edgecolor')
        if ax is None:
            fig, ax = plt.subplots()
        x, y = self.polygon.exterior.xy
        ax.plot(x, y, **plot_kwargs)
        for interior in self.polygon.interiors:
            xi, yi = interior.xy
            ax.plot(xi, yi, **plot_kwargs)
        ax.set_aspect('equal', 'box')
        return ax





@dataclass
class Trajectory:
    """An ordered list of robot poses forming a path."""
    poses: List[Pose]

    def draw(self,
             ax: Optional[plt.Axes] = None,
             show_robot: bool = False,
             car_length: float = 0.5,
             car_width: float = 0.2,
             **kwargs) -> plt.Axes:
        """
        Draw the trajectory path. Marks start and end with arrows.
        If show_robot=True, draws robot rectangle at start and end poses.
        """
        plot_kwargs = kwargs.copy()
        if 'edgecolor' in plot_kwargs:
            plot_kwargs['color'] = plot_kwargs.pop('edgecolor')
        if ax is None:
            fig, ax = plt.subplots()
        xs = [p[0] for p in self.poses]
        ys = [p[1] for p in self.poses]
        ax.plot(xs, ys, '-o', **plot_kwargs)
        # draw arrow at start
        x0, y0, yaw0 = self.poses[0]
        ax.arrow(x0, y0,
                 0.5 * math.cos(yaw0),
                 0.5 * math.sin(yaw0),
                 head_width=0.1, head_length=0.1, fc='g', ec='g')
        # draw arrow at end
        xe, ye, ya = self.poses[-1]
        ax.arrow(xe, ye,
                 0.5 * math.cos(ya),
                 0.5 * math.sin(ya),
                 head_width=0.1, head_length=0.1, fc='r', ec='r')
        # optionally draw robot footprint
        if show_robot:
            for pose, color in [(self.poses[0], 'green'), (self.poses[-1], 'red')]:
                corners = _robot_corners(pose[0], pose[1], pose[2], car_length, car_width)
                poly = plt.Polygon(corners, fill=False, edgecolor=color)
                ax.add_patch(poly)
                # mark head and tail
                hx = pose[0] + (car_length/2) * math.cos(pose[2])
                hy = pose[1] + (car_length/2) * math.sin(pose[2])
                tx = pose[0] - (car_length/2) * math.cos(pose[2])
                ty = pose[1] - (car_length/2) * math.sin(pose[2])
                ax.plot([tx, hx], [ty, hy], color=color, linewidth=2)
        ax.set_aspect('equal', 'box')
        return ax
    

    
    def envelope(self, car_length: float, car_width: float, buffer_factor: float = 0.0000000001) -> Envelope:
        # gather all footprint corners
        pts: List[XY] = []
        for x, y, yaw in self.poses:
            pts.extend(_robot_corners(x, y, yaw, car_length, car_width))
        hull = MultiPoint(pts).convex_hull
        half_diag = math.hypot(car_length, car_width) / 2.0
        buffered = hull.buffer(half_diag * buffer_factor)
        return Envelope(polygon=buffered)

@dataclass
class WallLine:
    """
    Represents open wall boundary with exit gap as MultiLineString.
    """
    geometry: MultiLineString

    def draw(self, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        color = kwargs.get('edgecolor', kwargs.get('color', 'black'))
        for seg in self.geometry.geoms:
            x,y = seg.xy
            ax.plot(x,y, color=color)
        ax.set_aspect('equal', 'box')
        return ax
    

@dataclass
class TargetPose:
    """
    Represents the target exit coordinates for training.
    """
    x: float
    y: float

    def to_tuple(self) -> XY:
        return (self.x, self.y)
    
    def draw(self, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        target_pose = self.to_tuple()
        ax.plot(target_pose[0], target_pose[1], 'ro', markersize=5, label='Target Exit')
        
        return ax

@dataclass
class SparseCopy:
    """
    Bundle of one sparse-copy configuration:
    - region:    the shifted Region
    - wall_line: the shifted WallLine with exit gap
    - start_pose: trajectory starting (x, y, yaw)
    - target_pose: exit target (x, y)
    """
    region: Region
    wall_line: WallLine
    start_pose: Pose
    target_pose: XY








def generate_subregions(car_length: float, car_width: float, total_maps: int) -> List[Region]:
    region_size = 10.0 * max(car_length, car_width)
    cols = math.ceil(math.sqrt(total_maps))
    rows = math.ceil(total_maps / cols)
    if cols * rows < total_maps:
        cols += 1
        rows = math.ceil(total_maps / cols)
    world_width = cols * region_size
    world_height = rows * region_size
    regions: List[Region] = []
    for i in range(cols):
        for j in range(rows):
            if len(regions) >= total_maps:
                break
            x_min = i * region_size - world_width / 2.0
            y_min = j * region_size - world_height / 2.0
            regions.append(Region(x_min, x_min + region_size, y_min, y_min + region_size))
    return regions


def compute_ackermann(v: float, delta: float, wheel_base: float) -> Tuple[float, float]:
    if abs(wheel_base) < 1e-6:
        raise ValueError("wheel_base must be non-zero")
    linear_velocity = v
    angular_velocity = v * math.tan(delta) / wheel_base if abs(delta) >= 1e-6 else 0.0
    return linear_velocity, angular_velocity


def _robot_corners(x: float, y: float, yaw: float, L: float, W: float) -> List[Tuple[float, float]]:
    hl, hw = L/2.0, W/2.0
    corners_local = [( hl,  hw), ( hl, -hw), (-hl, -hw), (-hl,  hw)]
    corners = []
    c = math.cos(yaw)
    s = math.sin(yaw)
    for lx, ly in corners_local:
        gx = x + lx * c - ly * s
        gy = y + lx * s + ly * c
        corners.append((gx, gy))
    return corners


def _inside_region(corners: List[Tuple[float, float]], region: Region) -> bool:
    for cx, cy in corners:
        if not (region.x_min <= cx <= region.x_max and region.y_min <= cy <= region.y_max):
            return False
    return True


def sample_trajectory(
    region: Region,
    wheel_base: float,
    car_length: float,
    car_width: float,
    n_steps: int = 50,
    dt: float = 0.1,
    beta: float = 0.05
) -> Trajectory:
    """
    Sample an Ackermann trajectory ensuring the robot stays within region.
    Uses midpoint integration for better accuracy.
    """
    v_max = 1.0
    delta_max = 37 * math.pi/180

    def sample_hard():
        v = random.choice([-1,1]) * random.uniform(0.25*v_max,0.75*v_max)
        delta = random.choice([-1,1]) * random.uniform(0.5*delta_max, delta_max)
        return v, delta

    def sample_simple():
        v_f = random.uniform(0.5*v_max, v_max)
        v_b = random.uniform(0.0, 0.5*v_max)
        v = v_f if random.random() < 0.9 else -v_b
        mag_d = random.uniform(0.0, 0.3*delta_max)
        delta = mag_d if random.random() < 0.9 else -mag_d
        return v, delta

    # initial pose
    x = random.uniform(region.x_min + 4*car_length, region.x_max - 4*car_length)
    y = random.uniform(region.y_min + 4*car_length, region.y_max - 4*car_length)
    yaw = random.uniform(0, 2*math.pi)
    poses: List[Pose] = [(x, y, yaw)]

    prev_v = prev_delta = 0.0
    for k in range(n_steps):
        alpha = k/(n_steps-1)
        w = math.exp(-beta*alpha)
  
        v, delta = (sample_hard() if random.random() < w else sample_simple())
        if random.random() < w:
            if v*prev_v > 0: v = -v
            if delta*prev_delta > 0: delta = -delta
        prev_v, prev_delta = v, delta


        # midpoint integration
        lv1, av1 = compute_ackermann(v, delta, wheel_base)
        yaw_mid = yaw + av1 * dt/2.0
        lv2 = v  # assume v constant over dt
        # update position with mid yaw
        x += lv2 * math.cos(yaw_mid) * dt
        y += lv2 * math.sin(yaw_mid) * dt
        # final yaw update
        yaw += av1 * dt

        # verify footprint inside region
        corners = _robot_corners(x, y, yaw, car_length, car_width)
        if not _inside_region(corners, region):
            break
        poses.append((x, y, yaw))
    return Trajectory(poses)




def generate_exit(
    envelope: Envelope,
    end_pose: Pose,
    car_length: float,
    car_width: float,
    gap_buffer: float = 0.02,
    step: float = 0.01,
    max_dist: float = 2.0
) -> Tuple[WallLine, TargetPose]:
    env_poly = envelope.polygon
    ext_line = LineString(env_poly.exterior.coords)
    x0,y0,yaw0 = end_pose
    # find exit point
    t=0.0
    while t<=max_dist:
        nx,ny = x0+t*math.cos(yaw0), y0+t*math.sin(yaw0)
        foot = Polygon(_robot_corners(nx,ny,yaw0,car_length,car_width+2*gap_buffer))
        if not env_poly.intersects(foot): break
        t+=step
    exit_x,exit_y = nx,ny
    target = TargetPose(exit_x,exit_y)
    # find intersection points
    strip = Polygon(_robot_corners(exit_x,exit_y,yaw0,car_length+20*gap_buffer,car_width+2*gap_buffer)).exterior
    inter = ext_line.intersection(strip)
    pts = []
    if hasattr(inter,'geoms'):
        for g in inter.geoms:
            if isinstance(g,Point): pts.append(g)
    elif isinstance(inter,Point): pts=[inter]
    if len(pts)<2:
        # fallback: no split
        return WallLine(geometry=MultiLineString([ext_line])), target
    if len(pts) > 2:
        # sort points by distance from exit point
        pts.sort(key=lambda p: math.hypot(p.x-exit_x, p.y-exit_y))
        pts = pts[:2]  # keep only the two closest points

    # create wall segments
    t1,t2=ext_line.project(pts[0]), ext_line.project(pts[1])
    tmin,tmax = min(t1,t2), max(t1,t2)
    if tmax-tmin < 1:
        seg1 = substring(ext_line,0,tmin)
        seg2 = substring(ext_line,tmax,ext_line.length)
        wall_geom = MultiLineString([seg1,seg2])
    else:
        seg1 = substring(ext_line,tmin,tmax)
        wall_geom = MultiLineString([seg1])
    return WallLine(geometry=wall_geom), target

def walls_from_wallline(
    wallline: WallLine,
    height: float,
    thickness: float,
    eps: float = 1e-6
) -> List[Wall]:
    """
    沿着 WallLine 上的每一个小折线段（相邻坐标对）生成一堵 Wall
    """
    walls: List[Wall] = []
    for seg in wallline.geometry.geoms:
        coords = list(seg.coords)
        # 对每条折线里的直线小段 (p[i] -> p[i+1])

        for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            # 如果这段太短，就跳过
            if length <= eps:
                continue
            length = math.hypot(x2-x1, y2-y1)
            yaw    = math.atan2(y2-y1, x2-x1)
            cx     = (x1 + x2)/2
            cy     = (y1 + y2)/2
            walls.append(Wall(
                length=length,
                thickness=thickness,
                height=height,
                pose=(cx, cy, yaw)
            ))
    return walls



def cylinders_from_wallline(
    wallline: WallLine,
    car_width: float,
    cylinder_radius: float,
    spacing: float
) -> List[Cylinder]:
    """
    沿着 WallLine 几何线段，等距采点，生成 Cylinder：
      - spacing 应小于车宽，保证没有缝隙
      - cylinder_radius 可设为 e.g. car_width/4
    """
    cylinders: List[Cylinder] = []
    for seg in wallline.geometry.geoms:
        L = seg.length
        # 沿着 seg 均匀采点
        n_pts = max(1, int(L // spacing))
        for i in range(n_pts + 1):
            pt = seg.interpolate(i / n_pts, normalized=True)
            cylinders.append(Cylinder(x=pt.x, y=pt.y, radius=cylinder_radius))

    return cylinders

import xml.etree.ElementTree as ET

def amend_sdf(sdf_root):
    # Find the world element
    world = sdf_root.find('world')
    
    # # Add environment element with GAZEBO_PLUGIN_PATH
    # env = ET.SubElement(world, 'environment')
    # ET.SubElement(env, 'GAZEBO_PLUGIN_PATH').text = '${GAZEBO_PLUGIN_PATH}:touch_plugin/build'
    
    # Create ROS plugin
    plugin = ET.Element('plugin', {'name': 'gazebo_ros_state', 'filename': 'libgazebo_ros_state.so'})
    ros = ET.SubElement(plugin, 'ros')
    namespace = ET.SubElement(ros, 'namespace')
    namespace.text = '/gazebo'
    update_rate = ET.SubElement(plugin, 'update_rate')
    update_rate.text = '1.0'
    world.append(plugin)
    
    # Create ground plane model
    model = ET.Element('model', {'name': 'ground_plane'})
    static = ET.SubElement(model, 'static')
    static.text = '1'
    
    link = ET.SubElement(model, 'link', {'name': 'link'})
    
    # Collision element
    collision = ET.SubElement(link, 'collision', {'name': 'collision'})
    collision_geom = ET.SubElement(collision, 'geometry')
    plane = ET.SubElement(collision_geom, 'plane')
    normal = ET.SubElement(plane, 'normal')
    normal.text = '0 0 1'
    size = ET.SubElement(plane, 'size')
    size.text = '100 100'
    
    surface = ET.SubElement(collision, 'surface')
    friction = ET.SubElement(surface, 'friction')
    ode = ET.SubElement(friction, 'ode')
    mu = ET.SubElement(ode, 'mu')
    mu.text = '100'
    mu2 = ET.SubElement(ode, 'mu2')
    mu2.text = '50'
    torsional = ET.SubElement(surface, 'torsional')
    torsional.append(ET.Element('ode'))
    contact = ET.SubElement(surface, 'contact')
    contact.append(ET.Element('ode'))
    ET.SubElement(surface, 'bounce')
    max_contacts = ET.SubElement(collision, 'max_contacts')
    max_contacts.text = '10'
    
    # Visual element
    visual = ET.SubElement(link, 'visual', {'name': 'visual'})
    cast_shadows = ET.SubElement(visual, 'cast_shadows')
    cast_shadows.text = '0'
    visual_geom = ET.SubElement(visual, 'geometry')
    v_plane = ET.SubElement(visual_geom, 'plane')
    v_normal = ET.SubElement(v_plane, 'normal')
    v_normal.text = '0 0 1'
    v_size = ET.SubElement(v_plane, 'size')
    v_size.text = '100 100'
    
    material = ET.SubElement(visual, 'material')
    script = ET.SubElement(material, 'script')
    uri = ET.SubElement(script, 'uri')
    uri.text = 'file://media/materials/scripts/gazebo.material'
    name = ET.SubElement(script, 'name')
    name.text = 'Gazebo/Grey'
    
    ET.SubElement(link, 'self_collide').text = '0'
    ET.SubElement(link, 'enable_wind').text = '0'
    ET.SubElement(link, 'kinematic').text = '0'
    
    world.append(model)
    
    # Create light
    light = ET.Element('light', {'name': 'sun', 'type': 'directional'})
    cast_shadows = ET.SubElement(light, 'cast_shadows')
    cast_shadows.text = '1'
    pose = ET.SubElement(light, 'pose')
    pose.text = '0 0 10 0 -0 0'
    diffuse = ET.SubElement(light, 'diffuse')
    diffuse.text = '0.8 0.8 0.8 1'
    specular = ET.SubElement(light, 'specular')
    specular.text = '0.2 0.2 0.2 1'
    
    attenuation = ET.SubElement(light, 'attenuation')
    range_elem = ET.SubElement(attenuation, 'range')
    range_elem.text = '1000'
    constant = ET.SubElement(attenuation, 'constant')
    constant.text = '0.9'
    linear = ET.SubElement(attenuation, 'linear')
    linear.text = '0.01'
    quadratic = ET.SubElement(attenuation, 'quadratic')
    quadratic.text = '0.001'
    
    direction = ET.SubElement(light, 'direction')
    direction.text = '-0.5 0.1 -0.9'
    
    spot = ET.SubElement(light, 'spot')
    inner_angle = ET.SubElement(spot, 'inner_angle')
    inner_angle.text = '0'
    outer_angle = ET.SubElement(spot, 'outer_angle')
    outer_angle.text = '0'
    falloff = ET.SubElement(spot, 'falloff')
    falloff.text = '0'
    
    world.append(light)
    
    # Create scene
    scene = ET.Element('scene')
    shadows = ET.SubElement(scene, 'shadows')
    shadows.text = '0'
    ambient = ET.SubElement(scene, 'ambient')
    ambient.text = '0.4 0.4 0.4 1'
    background = ET.SubElement(scene, 'background')
    background.text = '0.7 0.7 0.7 1'
    
    world.append(scene)
    
    return sdf_root



def main():
    car_length = 0.52
    car_width  = 0.48
    wheel_base = 0.513
    total_maps = 30

    # 1. 生成子区域
    regions = generate_subregions(car_length, car_width, total_maps=total_maps)

    # 2. 重新计算列数，与 generate_subregions 内部逻辑保持一致
    num_cols = math.ceil(math.sqrt(total_maps))
    n = len(regions)
    third = n // 3
    # 3. 绘图并标注 row/col/idx
    fig, ax = plt.subplots()
    for idx, region in enumerate(regions):
        row = idx // num_cols   # 整数除：第几行 (0-based)
        col = idx %  num_cols   # 取余：第几列 (0-based)
        region.draw(
            ax,
            edgecolor='blue',
            row=row+1, col=col+1, idx=idx+1  # +1 让编号从 1 开始
        )

    Trajs = []
    WallLines = []
    TargetPoses = []

    all_configs = []
    # Generate and draw one trajectory per region
    for idx, region in enumerate(regions):
        # 根据 idx 落在哪个区间，选择不同的参数
        if idx < third:
            # 第一段：最复杂/最困难
            traj_params = dict(n_steps=60, dt=0.1, beta=1.2)
            buf = 0.000000001
            gap = 0.03
        elif idx < 2*third:
            # 第二段：中等难度
            traj_params = dict(n_steps=50, dt=0.1, beta=0.8)
            buf = 0.000000001
            gap = 0.03
        else:
            # 第三段：最简单
            traj_params = dict(n_steps=30, dt=0.1, beta=0.2)
            buf = 0.000000001
            gap = 0.03

        # 3) 生成并绘制一条轨迹
        traj = sample_trajectory(
            region,
            wheel_base,
            car_length,
            car_width,
            **traj_params
        )
        traj.draw(ax, show_robot=True, car_length = car_length, car_width = car_width)
        Trajs.append(traj)
        env = traj.envelope(car_length, car_width, buffer_factor=buf)
        env.draw(ax, edgecolor='pink')
        wallLine,TargetPose = generate_exit(
            env,
            traj.poses[-1],car_length, car_width,
            gap_buffer=gap,
            step=0.01,max_dist=2.0)
        WallLines.append(wallLine)
        TargetPoses.append(TargetPose)
        wallLine.draw(ax, edgecolor='purple')
        TargetPose.draw(ax, color='blue', label='Target Exit')

        cfg_cont = Configuration(
        density='continuous',
        region=region,
        walls=walls_from_wallline(wallLine, height=0.6, thickness=0.01),
        cylinders=[],
        start_pose=traj.poses[0],
        target_position=TargetPose.to_tuple()
        )
        for w in cfg_cont.walls:
            w.draw(ax, edgecolor='orange',facecolor='orange', linewidth=0.5)
        all_configs.append(cfg_cont)

        # 5) 计算平移量：将第二份放在右侧，留 1 个 region_size 的间隔
    #    region_size = 20 * max(car_length, car_width)
    region_size = 10.0 * max(car_length, car_width)
    world_width = num_cols * region_size
    margin = region_size * 0.2
    dx = world_width + margin

    sparse_copies: List[SparseCopy] = []

    dx = world_width + margin  # 之前计算好的平移量

    for idx, (region, traj, wall, tgt) in enumerate(zip(regions, Trajs, WallLines, TargetPoses)):
        # 平移 region
        region2 = Region(
            x_min=region.x_min + dx,
            x_max=region.x_max + dx,
            y_min=region.y_min,
            y_max=region.y_max
        )
        # 平移 wall_line
        wall2 = WallLine(geometry=translate(wall.geometry, xoff=dx, yoff=0))
        # 平移 start_pose 和 target_pose
        x0, y0, yaw0 = traj.poses[0]
        start2 = (x0 + dx, y0, yaw0)
        tx, ty = tgt.to_tuple()
        target2 = (tx + dx, ty)

        sparse = SparseCopy(
            region=region2,
            wall_line=wall2,
            start_pose=start2,
            target_pose=target2
        )
        sparse_copies.append(sparse)

    for idx, sc in enumerate(sparse_copies, start=total_maps+1):
        sc.region.draw(ax, edgecolor='orange', idx=idx)
        sc.wall_line.draw(ax, edgecolor='red')
        # 绘制起点
        x0, y0, _ = sc.start_pose
        ax.scatter([x0], [y0], color='blue', s=20)
        # 绘制出口
        xt, yt = sc.target_pose
        ax.scatter([xt], [yt], color='green', s=30)
        ax.text(xt, yt, f"{idx}", color='green', va='bottom', ha='right', fontsize=8)
        cfg_sparse = Configuration(
            density='sparse',
            region=region,
            walls=[],
            cylinders=cylinders_from_wallline(
                sc.wall_line,
                car_width,
                cylinder_radius=0.02,
                spacing=0.15  # 小于车宽 0.28
            ),
            start_pose=sc.start_pose,
            target_position=sc.target_pose
        )
        for c in cfg_sparse.cylinders:
            c.draw(ax, edgecolor='black', facecolor='lightgray', linewidth=0.5)
        all_configs.append(cfg_sparse)





    print("Generated configurations:")
    for config in enumerate(all_configs, start=1):
        print(f"Config {config[0]}: {config[1].to_dict()}")

    ax.set_title(f'Trajectories in {n} Regions (×2: {2*total_maps})')

    # 在绘制完原始 + sparse_copy 之后：
    ax.relim()          # 重新计算所有 Artists 的数据范围
    ax.autoscale_view() # 自动缩放到合适的 limits
    plt.show()

    from generate_all_configs_world import build_full_world_sdf, write_sdf

    import os 
    
    sdf_root = build_full_world_sdf(all_configs)
    sdf_root = amend_sdf(sdf_root)
    # 写到文件
    output_sdfdir = "src/robot_gazebo/worlds"
    # Create directory if it doesn't exist
    os.makedirs(output_sdfdir, exist_ok=True)

    # Write SDF file with full path
    sdf_path = os.path.join(output_sdfdir, 'all_training.sdf')
    write_sdf(sdf_root, sdf_path)
    print("Wrote all_training.sdf")


    output_configs = "src/robot_gazebo/configs"
    # Create directory if it doesn't exist
    os.makedirs(output_configs, exist_ok=True)
    # Write JSON file with full path
    json_path = os.path.join(output_configs, 'configs.json')
    # all_configs 是你已经生成的 List[Configuration]
    with open(json_path, 'w') as f:
        json.dump([cfg.to_dict() for cfg in all_configs], f, indent=2)




    

if __name__ == '__main__':
    main()