"""Evaluation layouts for the LPX HI-VLA tasks.

These are plain benchmark initialization constants distilled from the LPX task
descriptions. They are pure evaluation constants.
"""

BIO2_LAYOUTS = {
    4: {
        "containers": ["C1", "C2", "C3", "C4", "C5"],
        "positions": {
            "C1": [0.0, -0.2, 0.065],
            "C2": [0.0, -0.1, 0.065],
            "C3": [0.0, 0.0, 0.065],
            "C4": [0.0, 0.1, 0.065],
            "C5": [0.0, 0.2, 0.065],
        },
        "red_pad_pos": [-0.25, -0.25, 0.003],
        "yellow_pad_pos": [-0.25, 0.25, 0.003],
    },
}

IND2_LAYOUTS = {
    6: {
        "num_boxes": 4,
        "box_positions": {
            "B1": [-0.05, -0.25, 0.05],
            "B2": [-0.05, -0.1, 0.05],
            "B3": [-0.1, 0.05, 0.05],
            "B4": [-0.05, 0.2, 0.05],
        },
    },
}

IND3_LAYOUTS = {
    1: {
        "sockets": ["S1", "S2", "S3", "S4", "S5"],
        "positions": {
            "S1": [-0.3, 0.3, 0.01],
            "S2": [-0.3, 0.15, 0.01],
            "S3": [-0.3, 0.0, 0.01],
            "S4": [-0.3, -0.15, 0.01],
            "S5": [-0.3, -0.3, 0.01],
        },
    },
}

IND5_LAYOUTS = {
    7: {
        "parts": ["P1", "P2", "P3"],
        "positions": {
            "P1": [-0.3, 0.25, 0.02],
            "P2": [-0.2, 0.35, 0.06],
            "P3": [-0.2, 0.25, 0.02],
        },
        "part_types": {
            "P1": "h_shape",
            "P2": "screw",
            "P3": "hex_prism",
        },
    },
}

IND5_TARGET_CENTER = [-0.5, 0.0]
IND5_TARGET_RADIUS = 0.08
