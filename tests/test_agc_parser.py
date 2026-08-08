import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "ingestion"))
from agc_parser import parse_agc_file  # noqa: E402

SAMPLE = os.path.join(
    os.path.dirname(__file__), "..", "data", "sample",
    "LUNAR_LANDING_GUIDANCE_EQUATIONS.agc"
)


def test_header_parsed():
    result = parse_agc_file(SAMPLE)
    assert result.header.get("Filename") == "LUNAR_LANDING_GUIDANCE_EQUATIONS.agc"
    assert result.header.get("Assembler") == "yaYUL"


def test_routines_extracted():
    result = parse_agc_file(SAMPLE)
    names = [r.name for r in result.routines]
    assert "RGVGCALC" in names
    assert "TTFINCR" in names


def test_instructions_have_fields():
    result = parse_agc_file(SAMPLE)
    routine = next(r for r in result.routines if r.name == "RGVGCALC")
    assert len(routine.instructions) > 0
    first = routine.instructions[0]
    assert first.opcode == "TC"
