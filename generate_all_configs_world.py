import math
import xml.etree.ElementTree as ET
from typing import List, Tuple

# 你的数据类
# from your_module import Configuration, Wall, Cylinder, Pose, XY
from config_types import Configuration, Wall, Cylinder, Pose, XY


def add_contact_sensor(link_elem, collision_name: str, sensor_name: str = 'sensor_contact',
                       update_rate: int = 1000):
    """
    在给定的 <link> 元素下，添加一个 contact 类型的 <sensor>，
    并让它监听名为 collision_name 的 <collision>。
    """
    sensor = ET.SubElement(link_elem, 'sensor',
                           {'name': sensor_name, 'type': 'contact'})
    # 下面这两个元素是可选的，你可以根据需要开启或关闭
    ET.SubElement(sensor, 'always_on').text   = 'true'
    ET.SubElement(sensor, 'update_rate').text = str(update_rate)
    contact = ET.SubElement(sensor, 'contact')
    # 这里要和 collision 的 name 对应
    ET.SubElement(contact, 'collision').text  = collision_name


def build_full_world_sdf(configs):
    sdf   = ET.Element('sdf',   {'version':'1.6'})
    world = ET.SubElement(sdf, 'world', {'name':'all_training'})
    # … ground_plane include, plugin 加载等 …    
    
    # Add touch plugin at the world level
    # touch_plugin = ET.SubElement(world, 'plugin', {
    #     'name': 'touch_plugin',
    #     'filename': 'libtouch_plugin.so'
    # })

    for idx, cfg in enumerate(configs, start=1):
        prefix = f'cfg{idx}'

        # —— 持续墙体 —— 
        for j, w in enumerate(cfg.walls, start=1):
            model = ET.SubElement(world, 'model', {'name':f'{prefix}_wall{j}'})
            ET.SubElement(model, 'static').text = 'true'
            x,y,yaw = w.pose
            ET.SubElement(model, 'pose').text = f'{x:.3f} {y:.3f} {w.height/2:.3f} 0 0 {yaw:.6f}'
            link = ET.SubElement(model, 'link', {'name':'link'})
            # collision
            col = ET.SubElement(link, 'collision', {'name':'col'})
            geom = ET.SubElement(col, 'geometry')
            box  = ET.SubElement(geom, 'box')
            ET.SubElement(box,'size').text = f'{w.length:.3f} {w.thickness:.3f} {w.height:.3f}'
            # visual
            vis = ET.SubElement(link, 'visual', {'name':'vis'})
            geom = ET.SubElement(vis, 'geometry')
            box  = ET.SubElement(geom, 'box')
            ET.SubElement(box,'size').text = f'{w.length:.3f} {w.thickness:.3f} {w.height:.3f}'
            # **加一个 contact sensor**
            add_contact_sensor(link, collision_name='col',
                               sensor_name='sensor_contact',
                               update_rate=1000)
            # plugin = ET.SubElement(model, 'plugin', {
            #     'filename':'libignition-gazebo-touchplugin-system.so',
            #     'name':'ignition::gazebo::systems::TouchPlugin'
            # })
            # ET.SubElement(plugin,'target').text = 'jetauto'
            # ET.SubElement(plugin,'namespace').text = 'cfg'
            # ET.SubElement(plugin,'time').text      = '0.001'
            # ET.SubElement(plugin,'enabled').text   = 'true'

        # —— 离散圆柱 —— 
        for j, c in enumerate(cfg.cylinders, start=1):
            model = ET.SubElement(world, 'model', {'name':f'{prefix}_cyl{j}'})
            ET.SubElement(model, 'static').text = 'true'
            # 圆柱底面放在地面上：z = radius
            ET.SubElement(model, 'pose').text = f'{c.x:.3f} {c.y:.3f} 0.3 0 0 0'
            link = ET.SubElement(model, 'link', {'name':'link'})
            # collision
            col = ET.SubElement(link, 'collision', {'name':'col'})
            geom = ET.SubElement(col, 'geometry')
            cyl  = ET.SubElement(geom, 'cylinder')
            ET.SubElement(cyl,'radius').text = f'{c.radius:.3f}'
            ET.SubElement(cyl,'length').text = f'0.6'
            # visual
            vis = ET.SubElement(link, 'visual', {'name':'vis'})
            geom = ET.SubElement(vis, 'geometry')
            cyl  = ET.SubElement(geom, 'cylinder')
            ET.SubElement(cyl,'radius').text = f'{c.radius:.3f}'
            ET.SubElement(cyl,'length').text = '0.6'
            # **同样加一个 contact sensor**
            add_contact_sensor(link, collision_name='col',
                               sensor_name='sensor_contact',
                               update_rate=1000)

            #（可选）在 world 里添加一个全局的 TouchPlugin，自动汇总所有 contact
            # plugin = ET.SubElement(model, 'plugin', {
            #     'filename':'libignition-gazebo-touchplugin-system.so',
            #     'name':'ignition::gazebo::systems::TouchPlugin'
            # })
            # ET.SubElement(plugin,'target').text = 'jetauto'
            # ET.SubElement(plugin,'namespace').text = 'cfg'
            # ET.SubElement(plugin,'time').text      = '0.001'
            # ET.SubElement(plugin,'enabled').text   = 'true'

        """     # 2c) 起点标记（绿色球）
        sx, sy, syaw = cfg.start_pose
        m = ET.SubElement(world, 'model', {'name':f'{prefix}_start'})
        ET.SubElement(m,'static').text = 'true'
        ET.SubElement(m,'pose').text = f'{sx:.3f} {sy:.3f} 0.1 0 0 {syaw:.6f}'
        link = ET.SubElement(m,'link',{'name':'link'})
        vis  = ET.SubElement(link,'visual',{'name':'vis'})
        geom = ET.SubElement(vis,'geometry')
        sph  = ET.SubElement(geom,'sphere')
        ET.SubElement(sph,'radius').text = '0.1'
        mat  = ET.SubElement(vis,'material')
        diff = ET.SubElement(mat,'diffuse')
        diff.text = '0 1 0 1'  # 绿

        # 2d) 目标标记（红色球），无朝向
        tx, ty = cfg.target_position
        m = ET.SubElement(world,'model',{'name':f'{prefix}_goal'})
        ET.SubElement(m,'static').text = 'true'
        ET.SubElement(m,'pose').text = f'{tx:.3f} {ty:.3f} 0.1 0 0 0'
        link = ET.SubElement(m,'link',{'name':'link'})
        vis  = ET.SubElement(link,'visual',{'name':'vis'})
        geom = ET.SubElement(vis,'geometry')
        sph  = ET.SubElement(geom,'sphere')
        ET.SubElement(sph,'radius').text = '0.1'
        mat  = ET.SubElement(vis,'material')
        diff = ET.SubElement(mat,'diffuse')
        diff.text = '1 0 0 1'  # 红 """


    return sdf

def write_sdf(element: ET.Element, path: str):
    tree = ET.ElementTree(element)
    tree.write(path, encoding='utf-8', xml_declaration=True)



