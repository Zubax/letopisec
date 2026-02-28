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

Open the API docs at the `/docs` endpoint in your browser, e.g., <http://localhost:8000/docs>.

Run behind a local gateway via Unix socket:

```bash
letopisec serve --uds /run/letopisec/letopisec.sock
```

Every runtime option can be configured with env vars like `LETOPISEC_HOST`/`.._PORT`, `LETOPISEC_DB_PATH`, etc.
CLI arguments override environment values.

### Store and retrieve CAN frames

Configure Zubax CANFace CF3D devices to use the endpoint where the server is running.
Existing dumps from CF3D memory cards can also be uploaded manually using [`letopisec_ingest.py`](tools/letopisec_ingest.py).

To retrieve data from the server, use [`letopisec_fetch.py`](tools/letopisec_fetch.py).

### Gateway integration

You can run the ASGI app through an external process manager/gateway using the app factory:

```bash
export LETOPISEC_DB_PATH=/var/lib/letopisec/letopisec.db
export LETOPISEC_LOG_FILE=/var/log/letopisec/server.log
export LETOPISEC_LOG_LEVEL=INFO
uvicorn --factory letopisec.server:create_app_from_env
```

## API endpoints

- `POST /cf3d/api/v1/commit` - Upload one or more binary CF3D records; the first HTTP response line is the cumulative ACK (`last_seqno`).
- `GET /cf3d/api/v1/devices` - List known devices with their latest heartbeat time (`last_heard_ts`) and last seen hardware UID (`last_uid`).
- `GET /cf3d/api/v1/boots` - List boot sessions for a device, including first/last record per boot, with optional commit-time window filtering.
- `GET /cf3d/api/v1/records` - Fetch records for a device and one or more boot IDs, with optional seqno bounds and long-polling for real-time streaming.

### Manual invocation

The REST API is very simple and can be exercised manually using wget or similar tools. Examples (`jq` is added for nicer formatting):

```bash
wget -qO- "http://localhost:8000/cf3d/api/v1/devices" | jq
```

```bash
wget -qO- "http://localhost:8000/cf3d/api/v1/boots?device=my+device" | jq
```

```bash
wget -qO- "http://localhost:8000/cf3d/api/v1/records?device=my+device&boot_id=1" | jq
```
