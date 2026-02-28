from __future__ import annotations

import logging
import struct
import unittest
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Annotated, ClassVar

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

from letopisec.database import Boot, Database, SqliteDatabase
from letopisec.fec_envelope import RECORD_BYTES, USER_DATA_BYTES, UnboxError, box, unbox
from letopisec.model import CANFrame, CANFrameRecord

app = FastAPI()
LOGGER = logging.getLogger(__name__)
_DATABASE = SqliteDatabase()


def _parse_device_uid(device_uid: Annotated[str, Query(min_length=1)]) -> int:
    try:
        return int(device_uid, 0)
    except ValueError as ex:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="device_uid must be an integer literal (auto-radix)",
        ) from ex


def get_database() -> Database:
    return _DATABASE


@app.post("/cf3d/api/v1/commit", response_class=PlainTextResponse)
async def commit(
    request: Request,
    device_uid: Annotated[int, Depends(_parse_device_uid)],
    device_tag: Annotated[str, Query(min_length=1)],
    database: Annotated[Database, Depends(get_database)],
) -> PlainTextResponse:
    """
    CAN frame record commit POST endpoint.
    The POST data is a binary blob consisting of RECORD_BYTES-large blobs, each blob is a Reed-Solomon encoded record.
    The server decodes each record and commits them into the storage, ignoring those that are already known,
    and returns the last known seqno such that the device could skip records that are already known to the server
    in the subsequent commit requests (similar to TCP cumulative ACK).
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
        last_seqno = database.commit(
            device_uid=device_uid,
            device_tag=device_tag,
            records=accepted_records,
        )
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
    status_code = status.HTTP_207_MULTI_STATUS if has_partial_failures else status.HTTP_200_OK
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
        status_code,
        last_seqno,
        len(accepted_records),
        decode_failures,
        parse_failures,
        full_records,
        trailing_bytes,
    )
    return PlainTextResponse(content="\n".join(body_lines), status_code=status_code)


def _parse_unboxed_commit_record(buf: bytes | bytearray | memoryview) -> CANFrameRecord | None:
    buf = memoryview(buf)
    if len(buf) < 1:
        raise ValueError("Empty buffer")

    version = int(buf[0])
    if version != 0:
        LOGGER.warning("Unsupported unboxed record version=%d", version)
        return None
    if len(buf) < 41:
        LOGGER.warning("Unboxed record too short for v0 header: len=%d", len(buf))
        return None

    boot_id, seqno, timestamp_us = struct.unpack_from("<QQQ", buf, 8)
    (can_id,) = struct.unpack_from("<I", buf, 36)
    data_len = int(buf[40])
    if data_len > 64:
        LOGGER.warning("Unboxed record has invalid CAN payload length: data_len=%d", data_len)
        return None

    total_len = 41 + data_len
    if len(buf) < total_len:
        LOGGER.warning("Unboxed record truncated: len=%d needed=%d", len(buf), total_len)
        return None

    data = bytes(buf[41:total_len])
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


class _FakeDatabase(Database):
    def __init__(self, ack_seqno: int = 0) -> None:
        self._ack_seqno = ack_seqno
        self.commits: list[tuple[int, str, list[CANFrameRecord]]] = []

    def commit(self, device_uid: int, device_tag: str, records: Sequence[CANFrameRecord]) -> int:
        self.commits.append((device_uid, device_tag, list(records)))
        return self._ack_seqno

    def get_device_tags(self) -> Iterable[str]:
        return []

    def get_boots(
        self, device_tag: str, earliest_commit: datetime | None, latest_commit: datetime | None
    ) -> Iterable[Boot]:
        return []

    def get_records(
        self, device_tag: str, boot_ids: Iterable[int], seqno_min: int | None, seqno_max: int | None
    ) -> Iterable[CANFrameRecord]:
        return []


class _RestAPITests(unittest.TestCase):
    client: ClassVar[TestClient]

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def setUp(self) -> None:
        self.database = _FakeDatabase(ack_seqno=321)
        app.dependency_overrides[get_database] = lambda: self.database

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

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

    def _post(self, payload: bytes = b"", *, device_uid: str = "123", device_tag: str = "abc"):
        return self.client.post(
            "/cf3d/api/v1/commit",
            params={"device_uid": device_uid, "device_tag": device_tag},
            content=payload,
        )

    def test_commit_empty_payload_returns_ack(self) -> None:
        response = self._post()
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
        response = self._post(payload, device_uid="0x10", device_tag="vehicle")
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
        response = self._post(payload)
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
        response = self._post(payload)
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
        response = self._post(payload)
        self.assertEqual(200, response.status_code)
        self.assertEqual("321", response.text)

        self.assertEqual(1, len(self.database.commits))
        _, _, committed_records = self.database.commits[0]
        self.assertEqual([valid], committed_records)

    def test_commit_all_invalid_records_returns_207_and_empty_commit(self) -> None:
        payload = (b"\x00" * RECORD_BYTES) + (b"\x00" * RECORD_BYTES)
        response = self._post(payload)
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
                response = self._post(device_uid=value)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
