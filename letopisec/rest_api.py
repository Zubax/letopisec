from __future__ import annotations

import asyncio
import logging
import struct
import unittest
from collections.abc import Iterable, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any, ClassVar
from unittest.mock import patch

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from letopisec.database import Boot, Database, SqliteDatabase
from letopisec.fec_envelope import RECORD_BYTES, USER_DATA_BYTES, UnboxError, box, unbox
from letopisec.model import CANFrame, CANFrameRecord

WAIT_MAX_TIMEOUT_S = 30
WAIT_POLL_INTERVAL_S = 0.25
RECORDS_DEFAULT_LIMIT = 1000
RECORDS_MAX_LIMIT = 10000

LOGGER = logging.getLogger(__name__)
router = APIRouter()


class ErrorResponse(BaseModel):
    detail: str | list[dict[str, Any]] = Field(description="Error details")


class CANFrameDTO(BaseModel):
    can_id: int = Field(description="CAN ID including SocketCAN flags")
    data_hex: str = Field(description="CAN frame payload bytes encoded as lowercase hexadecimal")


class CANFrameRecordDTO(BaseModel):
    ts_boot_us: int = Field(description="Microseconds from device boot at frame capture")
    boot_id: int = Field(description="Device boot identifier")
    seqno: int = Field(description="Monotonic frame sequence number")
    frame: CANFrameDTO


class BootDTO(BaseModel):
    boot_id: int
    first_record: CANFrameRecordDTO
    last_record: CANFrameRecordDTO


class DeviceTagsResponse(BaseModel):
    device_tags: list[str]


class BootsResponse(BaseModel):
    device_tag: str
    boots: list[BootDTO]


class RecordsFilterEcho(BaseModel):
    boot_ids: list[int]
    seqno_min: int | None
    seqno_max: int | None
    after_seqno: int | None
    wait_timeout_s: int
    limit: int
    offset: int


class RecordsResponse(BaseModel):
    device_tag: str
    filters: RecordsFilterEcho
    total_matched: int
    latest_seqno_seen: int | None
    timed_out: bool
    records: list[CANFrameRecordDTO]


def _serialize_frame(frame: CANFrame) -> CANFrameDTO:
    return CANFrameDTO(can_id=frame.can_id, data_hex=bytes(frame.data).hex())


def _serialize_record(record: CANFrameRecord) -> CANFrameRecordDTO:
    return CANFrameRecordDTO(
        ts_boot_us=record.ts_boot_us,
        boot_id=record.boot_id,
        seqno=record.seqno,
        frame=_serialize_frame(record.frame),
    )


def _serialize_boot(boot: Boot) -> BootDTO:
    return BootDTO(
        boot_id=boot.boot_id,
        first_record=_serialize_record(boot.first_record),
        last_record=_serialize_record(boot.last_record),
    )


def _parse_device_uid(
    device_uid: Annotated[str, Query(min_length=1, description="Integer literal (auto-radix)")],
) -> int:
    try:
        return int(device_uid, 0)
    except ValueError as ex:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="device_uid must be an integer literal (auto-radix)",
        ) from ex


def get_database(request: Request) -> Database:
    database = getattr(request.app.state, "database", None)
    if not isinstance(database, Database):
        LOGGER.critical(
            "Application database dependency is invalid: type=%s",
            None if database is None else type(database).__name__,
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="internal server error")
    return database


@router.post(
    "/cf3d/api/v1/commit",
    response_class=PlainTextResponse,
    tags=["commit"],
    summary="Commit CAN frame records",
    description=(
        "Upload one or more binary Reed-Solomon-wrapped CF3D records. "
        "Successful responses return cumulative ACK (last known seqno) as the first text line."
    ),
    responses={
        200: {
            "description": "All full records parsed/committed successfully.",
            "content": {"text/plain": {"example": "12345"}},
        },
        207: {
            "description": "Partial success (some records failed decode/parse).",
            "content": {"text/plain": {"example": "12345\naccepted=10 failed_decode=1 failed_parse=0"}},
        },
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def commit(
    request: Request,
    device_uid: Annotated[int, Depends(_parse_device_uid)],
    device_tag: Annotated[str, Query(min_length=1, description="Opaque device tag")],
    database: Annotated[Database, Depends(get_database)],
) -> PlainTextResponse:
    """
    Commit endpoint for CF3D devices.
    """
    payload = await request.body()
    full_records, trailing_bytes = divmod(len(payload), RECORD_BYTES)
    LOGGER.debug(
        "Commit request received: device_uid=%d device_tag=%r payload_bytes=%d full_records=%d trailing_bytes=%d",
        device_uid,
        device_tag,
        len(payload),
        full_records,
        trailing_bytes,
    )
    if trailing_bytes:
        LOGGER.warning(
            "Commit payload has trailing bytes that will be ignored: device_uid=%d device_tag=%r trailing_bytes=%d",
            device_uid,
            device_tag,
            trailing_bytes,
        )

    accepted_records: list[CANFrameRecord] = []
    decode_failures = 0
    parse_failures = 0
    for index in range(full_records):
        offset = index * RECORD_BYTES
        boxed = payload[offset : offset + RECORD_BYTES]
        unboxed = unbox(boxed)
        if isinstance(unboxed, UnboxError):
            decode_failures += 1
            LOGGER.error(
                "Commit record decode failed: device_uid=%d device_tag=%r index=%d error=%s",
                device_uid,
                device_tag,
                index,
                unboxed.name,
            )
            continue

        record = _parse_unboxed_commit_record(unboxed)
        if record is None:
            parse_failures += 1
            LOGGER.error(
                "Commit record parse failed after decode: device_uid=%d device_tag=%r index=%d",
                device_uid,
                device_tag,
                index,
            )
            continue

        accepted_records.append(record)
        LOGGER.debug(
            "Commit record accepted: device_uid=%d device_tag=%r index=%d seqno=%d boot_id=%d can_id=%d data_len=%d",
            device_uid,
            device_tag,
            index,
            record.seqno,
            record.boot_id,
            record.frame.can_id,
            len(bytes(record.frame.data)),
        )

    try:
        last_seqno = database.commit(device_uid=device_uid, device_tag=device_tag, records=accepted_records)
    except Exception:
        LOGGER.critical(
            "Unexpected commit exception: device_uid=%d device_tag=%r",
            device_uid,
            device_tag,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="internal server error",
        )

    has_partial_failures = decode_failures + parse_failures > 0
    response_status = status.HTTP_207_MULTI_STATUS if has_partial_failures else status.HTTP_200_OK
    body_lines = [str(last_seqno)]
    if has_partial_failures:
        body_lines.append(
            f"accepted={len(accepted_records)} failed_decode={decode_failures} failed_parse={parse_failures}"
        )

    LOGGER.info(
        "Commit request processed: device_uid=%d device_tag=%r status_code=%d last_seqno=%d "
        "accepted=%d failed_decode=%d failed_parse=%d full_records=%d trailing_bytes=%d",
        device_uid,
        device_tag,
        response_status,
        last_seqno,
        len(accepted_records),
        decode_failures,
        parse_failures,
        full_records,
        trailing_bytes,
    )
    return PlainTextResponse(content="\n".join(body_lines), status_code=response_status)


@router.get(
    "/cf3d/api/v1/device-tags",
    response_model=DeviceTagsResponse,
    tags=["query"],
    summary="List known device tags",
    description="Returns all known device tags currently present in the database.",
    responses={
        200: {
            "description": "Successful response",
            "content": {"application/json": {"example": {"device_tags": ["alpha", "beta"]}}},
        },
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
def get_device_tags(
    database: Annotated[Database, Depends(get_database)],
) -> DeviceTagsResponse:
    try:
        tags = list(database.get_device_tags())
    except Exception:
        LOGGER.critical("Unexpected exception while listing device tags", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="internal server error")

    LOGGER.info("Device tags query completed: count=%d", len(tags))
    return DeviceTagsResponse(device_tags=tags)


@router.get(
    "/cf3d/api/v1/boots",
    response_model=BootsResponse,
    tags=["query"],
    summary="Query boot ranges for a device tag",
    description=(
        "Returns boot ranges for the specified device tag. "
        "Commit timestamps are filtered using overlap semantics between earliest/latest bounds and boot commit span."
    ),
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {
                        "device_tag": "alpha",
                        "boots": [
                            {
                                "boot_id": 100,
                                "first_record": {
                                    "ts_boot_us": 10,
                                    "boot_id": 100,
                                    "seqno": 1,
                                    "frame": {"can_id": 291, "data_hex": "01"},
                                },
                                "last_record": {
                                    "ts_boot_us": 20,
                                    "boot_id": 100,
                                    "seqno": 2,
                                    "frame": {"can_id": 291, "data_hex": "02"},
                                },
                            }
                        ],
                    }
                }
            },
        },
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
def get_boots(
    device_tag: Annotated[str, Query(min_length=1, description="Device tag")],
    earliest_commit: Annotated[datetime | None, Query(description="Lower commit-time bound (ISO-8601)")] = None,
    latest_commit: Annotated[datetime | None, Query(description="Upper commit-time bound (ISO-8601)")] = None,
    database: Database = Depends(get_database),
) -> BootsResponse:
    LOGGER.debug(
        "Boots query request: device_tag=%r earliest_commit=%r latest_commit=%r",
        device_tag,
        earliest_commit,
        latest_commit,
    )
    try:
        boots = list(database.get_boots(device_tag, earliest_commit, latest_commit))
    except Exception:
        LOGGER.critical(
            "Unexpected exception while querying boots: device_tag=%r earliest_commit=%r latest_commit=%r",
            device_tag,
            earliest_commit,
            latest_commit,
            exc_info=True,
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="internal server error")

    serialized = [_serialize_boot(boot) for boot in boots]
    if not serialized:
        LOGGER.warning(
            "Boots query returned no results: device_tag=%r earliest_commit=%r latest_commit=%r",
            device_tag,
            earliest_commit,
            latest_commit,
        )
    else:
        LOGGER.info(
            "Boots query completed: device_tag=%r result_count=%d earliest_commit=%r latest_commit=%r",
            device_tag,
            len(serialized),
            earliest_commit,
            latest_commit,
        )
    return BootsResponse(device_tag=device_tag, boots=serialized)


def _resolve_effective_seqno_min(seqno_min: int | None, after_seqno: int | None) -> int | None:
    if after_seqno is None:
        return seqno_min
    cursor_min = after_seqno + 1
    if seqno_min is None:
        return cursor_min
    return max(seqno_min, cursor_min)


def _query_records_once(
    database: Database,
    device_tag: str,
    boot_ids: list[int],
    seqno_min: int | None,
    seqno_max: int | None,
) -> tuple[list[CANFrameRecord], int | None]:
    records = list(
        database.get_records(device_tag=device_tag, boot_ids=boot_ids, seqno_min=seqno_min, seqno_max=seqno_max)
    )
    records.sort(key=lambda record: record.seqno)
    latest_seqno_seen = records[-1].seqno if records else None
    return records, latest_seqno_seen


@router.get(
    "/cf3d/api/v1/records",
    response_model=RecordsResponse,
    tags=["query"],
    summary="Query CAN records",
    description=(
        "Returns records for specified device tag and boot IDs. "
        "Optional long polling is enabled via wait_timeout_s + after_seqno to wait for new records."
    ),
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {
                        "device_tag": "alpha",
                        "filters": {
                            "boot_ids": [1],
                            "seqno_min": 10,
                            "seqno_max": None,
                            "after_seqno": 9,
                            "wait_timeout_s": 0,
                            "limit": 1000,
                            "offset": 0,
                        },
                        "total_matched": 1,
                        "latest_seqno_seen": 10,
                        "timed_out": False,
                        "records": [
                            {
                                "ts_boot_us": 100,
                                "boot_id": 1,
                                "seqno": 10,
                                "frame": {"can_id": 291, "data_hex": "aabb"},
                            }
                        ],
                    }
                }
            },
        },
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_records(
    device_tag: Annotated[str, Query(min_length=1, description="Device tag")],
    boot_id: Annotated[list[int], Query(min_length=1, description="Repeated boot_id parameter")],
    seqno_min: Annotated[int | None, Query(description="Inclusive minimum sequence number")] = None,
    seqno_max: Annotated[int | None, Query(description="Inclusive maximum sequence number")] = None,
    after_seqno: Annotated[
        int | None, Query(description="Return records with seqno strictly greater than this")
    ] = None,
    wait_timeout_s: Annotated[
        int,
        Query(ge=0, le=WAIT_MAX_TIMEOUT_S, description="Optional long-poll timeout in seconds"),
    ] = 0,
    limit: Annotated[int, Query(ge=1, le=RECORDS_MAX_LIMIT, description="Page size")] = RECORDS_DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0, description="Pagination offset")] = 0,
    database: Database = Depends(get_database),
) -> RecordsResponse:
    boot_ids = sorted(set(int(value) for value in boot_id))
    effective_seqno_min = _resolve_effective_seqno_min(seqno_min, after_seqno)
    filters = RecordsFilterEcho(
        boot_ids=boot_ids,
        seqno_min=effective_seqno_min,
        seqno_max=seqno_max,
        after_seqno=after_seqno,
        wait_timeout_s=wait_timeout_s,
        limit=limit,
        offset=offset,
    )
    LOGGER.debug(
        "Records query request: device_tag=%r boot_ids=%s seqno_min=%r seqno_max=%r after_seqno=%r "
        "wait_timeout_s=%d limit=%d offset=%d",
        device_tag,
        boot_ids,
        effective_seqno_min,
        seqno_max,
        after_seqno,
        wait_timeout_s,
        limit,
        offset,
    )

    if seqno_max is not None and effective_seqno_min is not None and effective_seqno_min > seqno_max:
        LOGGER.warning(
            "Records query has invalid effective range; returning empty: device_tag=%r seqno_min=%d seqno_max=%d",
            device_tag,
            effective_seqno_min,
            seqno_max,
        )
        return RecordsResponse(
            device_tag=device_tag,
            filters=filters,
            total_matched=0,
            latest_seqno_seen=None,
            timed_out=False,
            records=[],
        )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_timeout_s
    poll_count = 0

    try:
        while True:
            poll_count += 1
            matching_records, latest_seqno_seen = _query_records_once(
                database=database,
                device_tag=device_tag,
                boot_ids=boot_ids,
                seqno_min=effective_seqno_min,
                seqno_max=seqno_max,
            )
            total_matched = len(matching_records)
            LOGGER.debug(
                "Records query poll result: device_tag=%r poll_count=%d total_matched=%d latest_seqno_seen=%r",
                device_tag,
                poll_count,
                total_matched,
                latest_seqno_seen,
            )

            if total_matched > 0:
                paged_records = matching_records[offset : offset + limit]
                serialized = [_serialize_record(record) for record in paged_records]
                LOGGER.info(
                    "Records query completed with data: device_tag=%r poll_count=%d total_matched=%d returned=%d",
                    device_tag,
                    poll_count,
                    total_matched,
                    len(serialized),
                )
                return RecordsResponse(
                    device_tag=device_tag,
                    filters=filters,
                    total_matched=total_matched,
                    latest_seqno_seen=latest_seqno_seen,
                    timed_out=False,
                    records=serialized,
                )

            now = loop.time()
            if wait_timeout_s <= 0 or now >= deadline:
                timed_out = wait_timeout_s > 0
                if timed_out:
                    LOGGER.warning(
                        "Records query timed out with no matching records: device_tag=%r wait_timeout_s=%d polls=%d",
                        device_tag,
                        wait_timeout_s,
                        poll_count,
                    )
                else:
                    LOGGER.info(
                        "Records query completed with empty snapshot: device_tag=%r poll_count=%d",
                        device_tag,
                        poll_count,
                    )
                return RecordsResponse(
                    device_tag=device_tag,
                    filters=filters,
                    total_matched=0,
                    latest_seqno_seen=latest_seqno_seen,
                    timed_out=timed_out,
                    records=[],
                )

            sleep_duration = min(WAIT_POLL_INTERVAL_S, deadline - now)
            LOGGER.debug(
                "Records query waiting for new records: device_tag=%r sleep_duration=%.3f poll_count=%d",
                device_tag,
                sleep_duration,
                poll_count,
            )
            await asyncio.sleep(sleep_duration)
    except Exception:
        LOGGER.critical(
            "Unexpected exception while querying records: device_tag=%r boot_ids=%s seqno_min=%r seqno_max=%r",
            device_tag,
            boot_ids,
            effective_seqno_min,
            seqno_max,
            exc_info=True,
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="internal server error")


def _parse_unboxed_commit_record(buf: bytes | bytearray | memoryview) -> CANFrameRecord | None:
    mv = memoryview(buf)
    if len(mv) < 1:
        LOGGER.warning("Unboxed record is empty")
        return None

    version = int(mv[0])
    if version != 0:
        LOGGER.warning("Unsupported unboxed record version=%d", version)
        return None
    if len(mv) < 41:
        LOGGER.warning("Unboxed record too short for v0 header: len=%d", len(mv))
        return None

    boot_id, seqno, timestamp_us = struct.unpack_from("<QQQ", mv, 8)
    (can_id,) = struct.unpack_from("<I", mv, 36)
    data_len = int(mv[40])
    if data_len > 64:
        LOGGER.warning("Unboxed record has invalid CAN payload length: data_len=%d", data_len)
        return None

    total_len = 41 + data_len
    if len(mv) < total_len:
        LOGGER.warning("Unboxed record truncated: len=%d needed=%d", len(mv), total_len)
        return None

    data = bytes(mv[41:total_len])
    return CANFrameRecord(
        ts_boot_us=int(timestamp_us),
        boot_id=int(boot_id),
        seqno=int(seqno),
        frame=CANFrame(can_id=int(can_id), data=data),
    )


def _pack_unboxed_commit_record_v0(record: CANFrameRecord) -> bytes:
    data = bytes(record.frame.data)
    if len(data) > 64:
        raise ValueError(f"CAN payload too long for v0 record: {len(data)} > 64")

    out = bytearray(USER_DATA_BYTES)
    out[0] = 0
    struct.pack_into("<QQQ", out, 8, record.boot_id, record.seqno, record.ts_boot_us)
    struct.pack_into("<I", out, 36, record.frame.can_id)
    out[40] = len(data)
    out[41 : 41 + len(data)] = data
    return bytes(out)


def _close_database(database: Database) -> None:
    close_method = getattr(database, "close", None)
    if not callable(close_method):
        LOGGER.debug("Database does not expose close(); skipping shutdown close for %s", type(database).__name__)
        return
    try:
        close_method()
        LOGGER.info("Database closed cleanly during app shutdown: type=%s", type(database).__name__)
    except Exception:
        LOGGER.error("Failed to close database during app shutdown: type=%s", type(database).__name__, exc_info=True)


def create_app(database: Database) -> FastAPI:
    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        try:
            yield
        finally:
            LOGGER.info("REST API app shutdown event received")
            _close_database(database)

    created_app = FastAPI(lifespan=_lifespan)
    created_app.state.database = database
    created_app.include_router(router)

    LOGGER.info("REST API app created with database backend=%s", type(database).__name__)
    return created_app


class _FakeDatabase(Database):
    def __init__(self, ack_seqno: int = 0) -> None:
        self._ack_seqno = ack_seqno
        self.commits: list[tuple[int, str, list[CANFrameRecord]]] = []
        self.device_tags: list[str] = []
        self.boots_by_tag: dict[str, list[Boot]] = {}
        self.records_by_tag: dict[str, list[CANFrameRecord]] = {}
        self.records_script_by_tag: dict[str, list[list[CANFrameRecord]]] = {}
        self.fail_methods: set[str] = set()
        self.last_get_boots_args: tuple[str, datetime | None, datetime | None] | None = None
        self.last_get_records_args: tuple[str, list[int], int | None, int | None] | None = None
        self.get_records_call_count = 0

    def commit(self, device_uid: int, device_tag: str, records: Sequence[CANFrameRecord]) -> int:
        if "commit" in self.fail_methods:
            raise RuntimeError("forced commit failure")
        self.commits.append((device_uid, device_tag, list(records)))
        return self._ack_seqno

    def get_device_tags(self) -> Iterable[str]:
        if "get_device_tags" in self.fail_methods:
            raise RuntimeError("forced device-tags failure")
        return list(self.device_tags)

    def get_boots(
        self, device_tag: str, earliest_commit: datetime | None, latest_commit: datetime | None
    ) -> Iterable[Boot]:
        if "get_boots" in self.fail_methods:
            raise RuntimeError("forced boots failure")
        self.last_get_boots_args = (device_tag, earliest_commit, latest_commit)
        return list(self.boots_by_tag.get(device_tag, []))

    def get_records(
        self, device_tag: str, boot_ids: Iterable[int], seqno_min: int | None, seqno_max: int | None
    ) -> Iterable[CANFrameRecord]:
        if "get_records" in self.fail_methods:
            raise RuntimeError("forced records failure")
        boot_list = [int(boot_id) for boot_id in boot_ids]
        self.last_get_records_args = (device_tag, boot_list, seqno_min, seqno_max)
        self.get_records_call_count += 1

        scripted = self.records_script_by_tag.get(device_tag)
        if scripted:
            source = list(scripted.pop(0))
        else:
            source = list(self.records_by_tag.get(device_tag, []))
        return self._filter_records(source, boot_list, seqno_min, seqno_max)

    @staticmethod
    def _filter_records(
        records: list[CANFrameRecord],
        boot_ids: list[int],
        seqno_min: int | None,
        seqno_max: int | None,
    ) -> list[CANFrameRecord]:
        if seqno_min is not None and seqno_max is not None and seqno_min > seqno_max:
            return []

        boot_id_set = set(int(boot_id) for boot_id in boot_ids)
        out: list[CANFrameRecord] = []
        for record in records:
            if record.boot_id not in boot_id_set:
                continue
            if seqno_min is not None and record.seqno < seqno_min:
                continue
            if seqno_max is not None and record.seqno > seqno_max:
                continue
            out.append(record)
        out.sort(key=lambda value: value.seqno)
        return out


class _RestAPITests(unittest.TestCase):
    app: ClassVar[FastAPI]
    client: ClassVar[TestClient]

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app(SqliteDatabase())
        cls.client = TestClient(cls.app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def setUp(self) -> None:
        self.database = _FakeDatabase(ack_seqno=321)
        self.app.dependency_overrides[get_database] = lambda: self.database

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()

    @staticmethod
    def _make_record(
        seqno: int = 1,
        *,
        boot_id: int = 1001,
        ts_boot_us: int = 5_000,
        can_id: int = 0x123,
        data: bytes = b"\x01\x02",
    ) -> CANFrameRecord:
        return CANFrameRecord(
            ts_boot_us=ts_boot_us,
            boot_id=boot_id,
            seqno=seqno,
            frame=CANFrame(can_id=can_id, data=data),
        )

    def _post_commit(self, payload: bytes = b"", *, device_uid: str = "123", device_tag: str = "abc"):
        return self.client.post(
            "/cf3d/api/v1/commit",
            params={"device_uid": device_uid, "device_tag": device_tag},
            content=payload,
        )

    def test_commit_empty_payload_returns_ack(self) -> None:
        response = self._post_commit()
        self.assertEqual(200, response.status_code)
        self.assertEqual("321", response.text)
        self.assertEqual(1, len(self.database.commits))
        committed_uid, committed_tag, committed_records = self.database.commits[0]
        self.assertEqual(123, committed_uid)
        self.assertEqual("abc", committed_tag)
        self.assertEqual([], committed_records)

    def test_commit_valid_record_is_decoded_and_committed(self) -> None:
        expected = self._make_record(seqno=42, boot_id=7, ts_boot_us=123456, can_id=0x1ABCDEFF, data=b"\xaa\xbb")
        payload = box(_pack_unboxed_commit_record_v0(expected))
        response = self._post_commit(payload, device_uid="0x10", device_tag="vehicle")
        self.assertEqual(200, response.status_code)
        self.assertEqual("321", response.text)

        self.assertEqual(1, len(self.database.commits))
        committed_uid, committed_tag, committed_records = self.database.commits[0]
        self.assertEqual(16, committed_uid)
        self.assertEqual("vehicle", committed_tag)
        self.assertEqual([expected], committed_records)

    def test_commit_partial_decode_failure_returns_207_with_ack_first_line(self) -> None:
        valid = self._make_record(seqno=5)
        payload = box(_pack_unboxed_commit_record_v0(valid)) + (b"\x00" * RECORD_BYTES)
        response = self._post_commit(payload)
        self.assertEqual(207, response.status_code)
        lines = response.text.splitlines()
        self.assertGreaterEqual(len(lines), 1)
        self.assertEqual("321", lines[0])

        self.assertEqual(1, len(self.database.commits))
        _, _, committed_records = self.database.commits[0]
        self.assertEqual([valid], committed_records)

    def test_commit_partial_parse_failure_returns_207_with_ack_first_line(self) -> None:
        malformed_unboxed = bytearray(USER_DATA_BYTES)
        malformed_unboxed[0] = 99  # Unsupported version.
        payload = box(bytes(malformed_unboxed))
        response = self._post_commit(payload)
        self.assertEqual(207, response.status_code)
        lines = response.text.splitlines()
        self.assertGreaterEqual(len(lines), 1)
        self.assertEqual("321", lines[0])

        self.assertEqual(1, len(self.database.commits))
        _, _, committed_records = self.database.commits[0]
        self.assertEqual([], committed_records)

    def test_commit_truncates_trailing_bytes_without_partial_status(self) -> None:
        valid = self._make_record(seqno=8)
        payload = box(_pack_unboxed_commit_record_v0(valid)) + b"trailing"
        response = self._post_commit(payload)
        self.assertEqual(200, response.status_code)
        self.assertEqual("321", response.text)

        self.assertEqual(1, len(self.database.commits))
        _, _, committed_records = self.database.commits[0]
        self.assertEqual([valid], committed_records)

    def test_commit_all_invalid_records_returns_207_and_empty_commit(self) -> None:
        payload = (b"\x00" * RECORD_BYTES) + (b"\x00" * RECORD_BYTES)
        response = self._post_commit(payload)
        self.assertEqual(207, response.status_code)
        lines = response.text.splitlines()
        self.assertGreaterEqual(len(lines), 1)
        self.assertEqual("321", lines[0])

        self.assertEqual(1, len(self.database.commits))
        _, _, committed_records = self.database.commits[0]
        self.assertEqual([], committed_records)

    def test_parse_unboxed_commit_record_v0_success(self) -> None:
        expected = self._make_record(seqno=777, boot_id=17, ts_boot_us=987654321, can_id=0x123, data=b"\x11\x22\x33")
        parsed = _parse_unboxed_commit_record(_pack_unboxed_commit_record_v0(expected))
        self.assertEqual(expected, parsed)

    def test_parse_unboxed_commit_record_rejects_unsupported_version(self) -> None:
        payload = bytearray(USER_DATA_BYTES)
        payload[0] = 1
        self.assertIsNone(_parse_unboxed_commit_record(payload))

    def test_parse_unboxed_commit_record_rejects_too_short_buffer(self) -> None:
        self.assertIsNone(_parse_unboxed_commit_record(b"\x00" * 40))

    def test_parse_unboxed_commit_record_rejects_data_length_over_64(self) -> None:
        payload = bytearray(200)
        payload[0] = 0
        payload[40] = 65
        self.assertIsNone(_parse_unboxed_commit_record(payload))

    def test_parse_unboxed_commit_record_rejects_truncated_payload(self) -> None:
        payload = bytearray(50)
        payload[0] = 0
        payload[40] = 32  # Needs at least 73 bytes.
        self.assertIsNone(_parse_unboxed_commit_record(payload))

    def test_device_uid_auto_radix_detection(self) -> None:
        committed_uids: list[int] = []
        for value in ("16", "0x10", "0o20", "0b10000"):
            with self.subTest(device_uid=value):
                response = self._post_commit(device_uid=value)
                self.assertEqual(200, response.status_code)
                self.assertEqual("321", response.text)
                committed_uids.append(self.database.commits[-1][0])
        self.assertEqual([16, 16, 16, 16], committed_uids)

    def test_missing_device_uid_is_rejected(self) -> None:
        response = self.client.post("/cf3d/api/v1/commit", params={"device_tag": "abc"})
        self.assertEqual(422, response.status_code)

    def test_missing_device_tag_is_rejected(self) -> None:
        response = self.client.post("/cf3d/api/v1/commit", params={"device_uid": "123"})
        self.assertEqual(422, response.status_code)

    def test_invalid_device_uid_is_rejected(self) -> None:
        response = self.client.post(
            "/cf3d/api/v1/commit",
            params={"device_uid": "xyz", "device_tag": "abc"},
        )
        self.assertEqual(422, response.status_code)

    def test_empty_device_tag_is_rejected(self) -> None:
        response = self.client.post(
            "/cf3d/api/v1/commit",
            params={"device_uid": "123", "device_tag": ""},
        )
        self.assertEqual(422, response.status_code)

    def test_get_device_tags_returns_json(self) -> None:
        self.database.device_tags = ["alpha", "beta"]
        response = self.client.get("/cf3d/api/v1/device-tags")
        self.assertEqual(200, response.status_code)
        self.assertEqual({"device_tags": ["alpha", "beta"]}, response.json())

    def test_get_device_tags_internal_error_returns_500(self) -> None:
        self.database.fail_methods.add("get_device_tags")
        response = self.client.get("/cf3d/api/v1/device-tags")
        self.assertEqual(500, response.status_code)

    def test_get_boots_returns_json(self) -> None:
        first = self._make_record(seqno=10, boot_id=7, ts_boot_us=1, can_id=0x100, data=b"\xaa")
        last = self._make_record(seqno=20, boot_id=7, ts_boot_us=2, can_id=0x200, data=b"\xbb")
        self.database.boots_by_tag["alpha"] = [Boot(boot_id=7, first_record=first, last_record=last)]

        response = self.client.get("/cf3d/api/v1/boots", params={"device_tag": "alpha"})
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual("alpha", body["device_tag"])
        self.assertEqual(1, len(body["boots"]))
        self.assertEqual("aa", body["boots"][0]["first_record"]["frame"]["data_hex"])
        self.assertEqual("bb", body["boots"][0]["last_record"]["frame"]["data_hex"])

    def test_get_boots_unknown_tag_returns_empty_list(self) -> None:
        response = self.client.get("/cf3d/api/v1/boots", params={"device_tag": "unknown"})
        self.assertEqual(200, response.status_code)
        self.assertEqual({"device_tag": "unknown", "boots": []}, response.json())

    def test_get_boots_passes_parsed_datetime_filters(self) -> None:
        response = self.client.get(
            "/cf3d/api/v1/boots",
            params={
                "device_tag": "alpha",
                "earliest_commit": "2024-01-01T00:00:00Z",
                "latest_commit": "2024-01-01T01:00:00Z",
            },
        )
        self.assertEqual(200, response.status_code)
        self.assertIsNotNone(self.database.last_get_boots_args)
        assert self.database.last_get_boots_args is not None
        _, earliest, latest = self.database.last_get_boots_args
        self.assertIsNotNone(earliest)
        self.assertIsNotNone(latest)
        assert earliest is not None and latest is not None
        self.assertEqual(int(datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp()), int(earliest.timestamp()))
        self.assertEqual(int(datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc).timestamp()), int(latest.timestamp()))

    def test_get_boots_invalid_datetime_rejected(self) -> None:
        response = self.client.get("/cf3d/api/v1/boots", params={"device_tag": "alpha", "earliest_commit": "bad"})
        self.assertEqual(422, response.status_code)

    def test_get_records_requires_boot_id(self) -> None:
        response = self.client.get("/cf3d/api/v1/records", params={"device_tag": "alpha"})
        self.assertEqual(422, response.status_code)

    def test_get_records_returns_paginated_results(self) -> None:
        self.database.records_by_tag["alpha"] = [
            self._make_record(seqno=1, boot_id=1, data=b"\x01"),
            self._make_record(seqno=2, boot_id=1, data=b"\x02"),
            self._make_record(seqno=3, boot_id=1, data=b"\x03"),
        ]

        response = self.client.get(
            "/cf3d/api/v1/records",
            params={"device_tag": "alpha", "boot_id": 1, "limit": 2, "offset": 1},
        )
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual(3, body["total_matched"])
        self.assertEqual(2, len(body["records"]))
        self.assertEqual([2, 3], [item["seqno"] for item in body["records"]])
        self.assertFalse(body["timed_out"])

    def test_get_records_unknown_tag_returns_empty(self) -> None:
        response = self.client.get("/cf3d/api/v1/records", params={"device_tag": "unknown", "boot_id": 1})
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual(0, body["total_matched"])
        self.assertEqual([], body["records"])
        self.assertFalse(body["timed_out"])

    def test_get_records_invalid_range_returns_empty_200(self) -> None:
        response = self.client.get(
            "/cf3d/api/v1/records",
            params={"device_tag": "alpha", "boot_id": 1, "seqno_min": 10, "seqno_max": 1},
        )
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual(0, body["total_matched"])
        self.assertEqual([], body["records"])
        self.assertFalse(body["timed_out"])

    def test_get_records_after_seqno_filters_results(self) -> None:
        self.database.records_by_tag["alpha"] = [
            self._make_record(seqno=1, boot_id=1),
            self._make_record(seqno=2, boot_id=1),
            self._make_record(seqno=3, boot_id=1),
        ]
        response = self.client.get(
            "/cf3d/api/v1/records",
            params={"device_tag": "alpha", "boot_id": 1, "after_seqno": 2},
        )
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual(1, body["total_matched"])
        self.assertEqual([3], [item["seqno"] for item in body["records"]])

    def test_get_records_long_poll_wakes_when_new_data_appears(self) -> None:
        wake_record = self._make_record(seqno=9, boot_id=1)
        self.database.records_script_by_tag["alpha"] = [[], [wake_record]]

        with patch("letopisec.rest_api.WAIT_POLL_INTERVAL_S", 0.01):
            response = self.client.get(
                "/cf3d/api/v1/records",
                params={"device_tag": "alpha", "boot_id": 1, "after_seqno": 8, "wait_timeout_s": 1},
            )
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertFalse(body["timed_out"])
        self.assertEqual([9], [item["seqno"] for item in body["records"]])
        self.assertGreaterEqual(self.database.get_records_call_count, 2)

    def test_get_records_long_poll_timeout_returns_timed_out_true(self) -> None:
        self.database.records_script_by_tag["alpha"] = [[], [], [], []]

        with patch("letopisec.rest_api.WAIT_POLL_INTERVAL_S", 0.01):
            response = self.client.get(
                "/cf3d/api/v1/records",
                params={"device_tag": "alpha", "boot_id": 1, "after_seqno": 100, "wait_timeout_s": 1},
            )
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertTrue(body["timed_out"])
        self.assertEqual([], body["records"])

    def test_get_records_wait_timeout_validation(self) -> None:
        response = self.client.get(
            "/cf3d/api/v1/records",
            params={"device_tag": "alpha", "boot_id": 1, "wait_timeout_s": 31},
        )
        self.assertEqual(422, response.status_code)

    def test_get_records_internal_error_returns_500(self) -> None:
        self.database.fail_methods.add("get_records")
        response = self.client.get("/cf3d/api/v1/records", params={"device_tag": "alpha", "boot_id": 1})
        self.assertEqual(500, response.status_code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
