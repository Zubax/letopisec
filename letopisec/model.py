from dataclasses import dataclass


@dataclass(frozen=True)
class CANFrameRecord:
    ts_boot_us: int
    """Time in microseconds from device bootup to frame capture."""
    boot_id: int
    """Sequential boot count; incremented each bootup."""
    seqno: int
    """Unique sequential number of the frame on this device across reboots."""
    frame: "CANFrame"


@dataclass(frozen=True)
class CANFrame:
    can_id: int
    data: bytes | bytearray
