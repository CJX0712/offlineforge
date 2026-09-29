"""OfflineForge 错误体系 — 与 forge 系列统一的错误码约定."""


class OfflineForgeError(Exception):
    """OfflineForge 基类错误 (E000)."""

    code = "E000"

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        self.message = message
        self.cause = cause
        super().__init__(f"[{self.code}] {message}")


class DataError(OfflineForgeError):
    """数据集/样本不合法 (E100)."""

    code = "E100"


class ConfigError(OfflineForgeError):
    """配置项非法 (E200)."""

    code = "E200"


class BackendUnavailable(OfflineForgeError):
    """可选后端 (d3rlpy/torch) 不可用 (E300)."""

    code = "E300"


class NumericalError(OfflineForgeError):
    """数值不收敛/非有限 (E400)."""

    code = "E400"


class PipelineError(OfflineForgeError):
    """流水线编排错误 (E500)."""

    code = "E500"


__all__ = [
    "OfflineForgeError",
    "DataError",
    "ConfigError",
    "BackendUnavailable",
    "NumericalError",
    "PipelineError",
]
