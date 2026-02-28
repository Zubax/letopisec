from __future__ import annotations

import unittest
from typing import Annotated, ClassVar

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient

app = FastAPI()


def _parse_device_uid(device_uid: Annotated[str, Query(min_length=1)]) -> int:
    try:
        return int(device_uid, 0)
    except ValueError as ex:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="device_uid must be an integer literal (auto-radix)",
        ) from ex


@app.post("/cf3d/api/v1/commit", response_class=PlainTextResponse)
def commit(
    device_uid: Annotated[int, Depends(_parse_device_uid)],
    device_tag: Annotated[str, Query(min_length=1)],
) -> str:
    return "0"


class _RestAPITests(unittest.TestCase):
    client: ClassVar[TestClient]

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_commit_returns_plain_zero(self) -> None:
        response = self.client.post(
            "/cf3d/api/v1/commit",
            params={"device_uid": "123", "device_tag": "abc"},
        )
        self.assertEqual(200, response.status_code)
        self.assertEqual("0", response.text)

    def test_device_uid_auto_radix_detection(self) -> None:
        for value in ("16", "0x10", "0o20", "0b10000"):
            with self.subTest(device_uid=value):
                response = self.client.post(
                    "/cf3d/api/v1/commit",
                    params={"device_uid": value, "device_tag": "abc"},
                )
                self.assertEqual(200, response.status_code)
                self.assertEqual("0", response.text)

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
