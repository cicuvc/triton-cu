from __future__ import annotations

from enum import Enum


class TranslatorTarget(str, Enum):
    """Target architecture for the Triton-to-Gluon translator."""

    GENERIC = "generic"
    SM80 = "sm80"
    SM90 = "sm90"
    SM100 = "sm100"
    SM103 = "sm103"

    @classmethod
    def _missing_(cls, value: object) -> "TranslatorTarget | None":
        if value not in cls._value2member_map_:
            return None
        if isinstance(value, str):
            return cls(value)
        return None

    @property
    def is_nvidia(self) -> bool:
        return True

    @property
    def tensor_descriptor_import(self) -> str:
        module = "nvidia.hopper.tma"
        return f"from triton.experimental.gluon.language.{module} import tensor_descriptor"

    @property
    def helpers_module(self) -> str:
        base = "triton.tools.triton_to_gluon_translator"

        if self in (TranslatorTarget.SM100, TranslatorTarget.SM103):
            return f"{base}.blackwell_helpers"

        if self in (TranslatorTarget.SM90):
            return f"{base}.hopper_helpers"

        if self in (TranslatorTarget.SM80):
            return f"{base}.nvidia_helpers"

        return f"{base}.common_helpers"
