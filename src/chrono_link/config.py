"""Strict configuration and runtime validation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from chrono_link.contracts import BandSpec


class ConfigError(ValueError):
    """Configuration is invalid or incompatible with runtime metadata."""


DEFAULT_FILTER_BANDS = (
    BandSpec("broadband", 1.0, 40.0),
    BandSpec("mi", 8.0, 30.0),
    BandSpec("mu", 8.0, 12.0),
    BandSpec("beta", 13.0, 30.0),
)

ENV_KEYS = {
    "CHRONO_BOARD",
    "CHRONO_SERIAL_PORT",
    "CHRONO_MAINS_HZ",
    "CHRONO_OUTPUT_ROOT",
}


@dataclass(frozen=True)
class BoardConfig:
    profile: str = "synthetic"
    serial_port: str | None = None
    decode_channels: tuple[str, ...] = ("C3", "C4")
    record_channels: tuple[str, ...] | str = "all"

    def __post_init__(self) -> None:
        if self.profile not in {"synthetic", "cyton"}:
            raise ConfigError(f"unsupported board profile: {self.profile}")
        if not self.decode_channels or len(set(self.decode_channels)) != len(self.decode_channels):
            raise ConfigError("decode_channels must be non-empty and unique")
        if self.record_channels != "all":
            if not self.record_channels or len(set(self.record_channels)) != len(
                self.record_channels
            ):
                raise ConfigError("record_channels must be 'all' or a non-empty unique tuple")
            missing = set(self.decode_channels) - set(self.record_channels)
            if missing:
                raise ConfigError(
                    f"record_channels must include decode channels: {sorted(missing)}"
                )
        # A Cyton request can be constructed and inspected without hardware. The
        # serial-port requirement is enforced immediately before a session is
        # acquired by BrainFlowSource.prepare().


@dataclass(frozen=True)
class SignalConfig:
    window_samples: int = 250
    hop_samples: int = 64
    warmup_samples: int = 1000
    mains_hz: int = 50
    notch_q: float = 30.0
    dc_highpass_hz: float = 0.5
    flat_abs_tol: float = 1e-12
    flat_rel_tol: float = 1e-8
    filter_bands: tuple[BandSpec, ...] = DEFAULT_FILTER_BANDS
    vector_feature_bands: tuple[str, ...] = ("mu", "beta")

    def __post_init__(self) -> None:
        if min(self.window_samples, self.hop_samples, self.warmup_samples) <= 0:
            raise ConfigError("window, hop, and warmup samples must be positive")
        if self.mains_hz not in {50, 60}:
            raise ConfigError("mains_hz must be 50 or 60")
        if self.notch_q <= 0 or self.dc_highpass_hz <= 0:
            raise ConfigError("filter frequencies and Q must be positive")
        names = tuple(band.name for band in self.filter_bands)
        if len(set(names)) != len(names):
            raise ConfigError("filter band names must be unique")
        missing = set(self.vector_feature_bands) - set(names)
        if missing:
            raise ConfigError(f"vector feature bands are not filtered: {sorted(missing)}")

    def validate_for(self, fs: int) -> None:
        if fs <= 0:
            raise ConfigError("sampling rate must be positive")
        nyquist = fs / 2
        if self.dc_highpass_hz >= nyquist:
            raise ConfigError(f"dc_highpass_hz must be below Nyquist ({nyquist:g} Hz)")
        for band in self.filter_bands:
            if band.high_hz >= nyquist:
                raise ConfigError(
                    f"band {band.name} high edge {band.high_hz:g} must be below "
                    f"Nyquist ({nyquist:g} Hz)"
                )


@dataclass(frozen=True)
class ChronoConfig:
    board: BoardConfig = field(default_factory=BoardConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    output_root: Path = Path(".")

    @classmethod
    def synthetic(cls) -> ChronoConfig:
        return cls()

    @classmethod
    def from_sources(
        cls,
        json_path: Path | None = None,
        cli: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> ChronoConfig:
        data: dict[str, Any] = {}
        if json_path is not None:
            try:
                loaded = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigError(f"cannot read config {json_path}: {exc}") from exc
            if not isinstance(loaded, dict):
                raise ConfigError("configuration root must be an object")
            data = loaded
        unknown = set(data) - {"board", "signal", "output_root"}
        if unknown:
            raise ConfigError(f"unknown configuration keys: {sorted(unknown)}")

        board_data = dict(data.get("board", {}))
        signal_data = dict(data.get("signal", {}))
        output_root = data.get("output_root", ".")
        env = dict(os.environ if environ is None else environ)
        unknown_env = {key for key in env if key.startswith("CHRONO_")} - ENV_KEYS
        if unknown_env:
            raise ConfigError(f"unknown Chrono environment keys: {sorted(unknown_env)}")
        if "CHRONO_BOARD" in env:
            board_data["profile"] = env["CHRONO_BOARD"]
        if "CHRONO_SERIAL_PORT" in env:
            board_data["serial_port"] = env["CHRONO_SERIAL_PORT"]
        if "CHRONO_MAINS_HZ" in env:
            signal_data["mains_hz"] = int(env["CHRONO_MAINS_HZ"])
        if "CHRONO_OUTPUT_ROOT" in env:
            output_root = env["CHRONO_OUTPUT_ROOT"]

        cli_data = {key: value for key, value in (cli or {}).items() if value is not None}
        for key in ("profile", "serial_port", "decode_channels", "record_channels"):
            if key in cli_data:
                board_data[key] = cli_data[key]
        if "mains_hz" in cli_data:
            signal_data["mains_hz"] = cli_data["mains_hz"]
        if "output_root" in cli_data:
            output_root = cli_data["output_root"]

        allowed_board = {field.name for field in BoardConfig.__dataclass_fields__.values()}
        allowed_signal = {field.name for field in SignalConfig.__dataclass_fields__.values()}
        if set(board_data) - allowed_board:
            raise ConfigError(f"unknown board keys: {sorted(set(board_data) - allowed_board)}")
        if set(signal_data) - allowed_signal:
            raise ConfigError(f"unknown signal keys: {sorted(set(signal_data) - allowed_signal)}")

        if isinstance(board_data.get("decode_channels"), list):
            board_data["decode_channels"] = tuple(board_data["decode_channels"])
        if isinstance(board_data.get("record_channels"), list):
            board_data["record_channels"] = tuple(board_data["record_channels"])
        if "filter_bands" in signal_data:
            signal_data["filter_bands"] = tuple(
                BandSpec(**item) if isinstance(item, dict) else item
                for item in signal_data["filter_bands"]
            )
        if isinstance(signal_data.get("vector_feature_bands"), list):
            signal_data["vector_feature_bands"] = tuple(signal_data["vector_feature_bands"])

        return cls(
            board=BoardConfig(**board_data),
            signal=SignalConfig(**signal_data),
            output_root=Path(output_root),
        )

    def with_board(self, **changes: Any) -> ChronoConfig:
        return replace(self, board=replace(self.board, **changes))

    def canonical_json(self) -> str:
        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
