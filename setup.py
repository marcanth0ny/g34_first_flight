from setuptools import setup

package_name = 'g34_first_flight'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/g34_first_flight.launch.py']),
        ('share/' + package_name + '/params', ['params/g34_first_flight_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yo',
    maintainer_email='you@example.com',
    description='PX4 offboard first hop + altitude & attitude tuning sequences',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'g34_first_flight_node = g34_first_flight.first_flight_node:main',
        ],
    },
)
