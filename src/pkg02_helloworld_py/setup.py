from setuptools import find_packages, setup

package_name = 'pkg02_helloworld_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xxd',
    maintainer_email='xxd@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera = pkg02_helloworld_py.camera_catch:main',
            'camera_node = pkg02_helloworld_py.camera_node:main',
            'sub_openvino_blue = pkg02_helloworld_py.sub_openvino_blue:main',
            'serial_send = pkg02_helloworld_py.serial_send:main',
            'sub_openvino_UKF = pkg02_helloworld_py.sub_openvino_UKF:main',
            'Grabimage = pkg02_helloworld_py.Grabimage:main',
            'sub_openvino_red = pkg02_helloworld_py.sub_openvino_red:main',
            'sub = pkg02_helloworld_py.sub:main',
        ],
    },
)
