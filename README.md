# Letopisec -- Data collection server for CANFace CF3D

For details, please refer to the [requirements document](https://forum.zubax.com/t/cf3d-doordash-datalogger-functional-requirements/2802).

This is the server part: a Python REST service invoked by CF3D data loggers when the server is reachable. Uploaded CAN frames are stored in SQLite.

## Usage

Install from this directory:

```bash
pip install .  # optionally use -e . for editable installation
```

Run the server:

```bash
letopisec serve  # See --help for extra info.
```

Run behind a local gateway via Unix socket:

```bash
letopisec serve --uds /run/letopisec/letopisec.sock
```

Every runtime option can be configured with env vars like `LETOPISEC_HOST`/`.._PORT`, `LETOPISEC_DB_PATH`, etc.
CLI arguments override environment values.

### Gateway integration

You can run the ASGI app through an external process manager/gateway using the app factory:

```bash
export LETOPISEC_DB_PATH=/var/lib/letopisec/letopisec.db
export LETOPISEC_LOG_FILE=/var/log/letopisec/server.log
export LETOPISEC_LOG_LEVEL=INFO
uvicorn --factory letopisec.server:create_app_from_env
```
