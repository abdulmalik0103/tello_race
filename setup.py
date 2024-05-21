from setuptools import setup
import os
from glob import glob

package_name = 'tello_race'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share/' + package_name, 'launch/'),
         glob('launch/*.[pxy][yma]*')),
        (os.path.join('share/' + package_name, 'worlds/'),
         glob('worlds/*.world*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abdul',
    maintainer_email='abmali@utu.fi',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = tello_race.detector:main',
            'controller= tello_race.controller:main',
        ],
    },
)
