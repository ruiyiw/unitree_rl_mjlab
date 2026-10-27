"""Unchanged disturbance grids from the reference stepping evaluator."""

from .evaluate import conditions

CHANNELS = (
    "tilt",
    "speed",
    "aligned",
    "left_transverse",
    "opposing",
    "right_transverse",
)
STAGES = ((10.0, 0.3), (12.5, 0.375), (15.0, 0.45), (17.5, 0.525), (20.0, 0.6))


def stepping_conditions(stages):
    cases = conditions(True)
    baseline = cases[0]
    result = [baseline]
    for prefix, scale in (("original", 1), ("doubled", 2)):
        for case in cases[1:]:
            tilt, speed = case["tilt_deg"] * scale, case["speed"] * scale
            result.append(
                {
                    **case,
                    "name": f"{prefix}_{case['kind']}_{tilt:g}deg_{speed:g}mps_rel{case['relative_deg']}",
                    "tilt_deg": tilt,
                    "speed": speed,
                }
            )
    for channel, stage in enumerate(stages):
        tilt, speed = STAGES[stage]
        result.append(
            {
                "name": f"frontier_{CHANNELS[channel]}",
                "kind": "tilt"
                if channel == 0
                else "velocity"
                if channel == 1
                else "combined",
                "tilt_deg": 0 if channel == 1 else tilt,
                "speed": 0 if channel == 0 else speed,
                "relative_deg": 0 if channel < 2 else (channel - 2) * 90,
            }
        )
    return result
