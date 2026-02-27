# Letopisec -- Data collection server for CANFace CF3D

For details, please refer to the [requirements document](https://forum.zubax.com/t/cf3d-doordash-datalogger-functional-requirements/2802).

This is the server part: a simple Python script that serves a REST endpoint that is invoked by CF3D data loggers when the server is reachable, uploading all CAN frames connected while the device was offline. The server stores all uploaded frames into an sqlite database.
