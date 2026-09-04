"""Frozen 25-class order and competition class groups."""

CLASS_NAMES = (
    "HM",
    "LQS",
    "QHS",
    "MS",
    "A1_SU-35",
    "A2_C-130",
    "A3_C-17",
    "A4_C-5",
    "A5_F-16",
    "A6_TU-160",
    "A7_E-3",
    "A8_B-52",
    "A9_P-3C",
    "A10_B-1B",
    "A11_E-8",
    "A12_TU-22",
    "A13_F-15",
    "A14_KC-135",
    "A15_F-22",
    "A16_FA-18",
    "A17_TU-95",
    "A18_KC-10",
    "A19_SU-34",
    "A20_SU-24",
    "FSC",
)

CLASS_COUNT = len(CLASS_NAMES)
SHIP_CLASS_IDS = frozenset(range(0, 4))
AIRCRAFT_CLASS_IDS = frozenset(range(4, 24))
VEHICLE_CLASS_IDS = frozenset({24})
GROUP_CLASS_IDS = {
    "ship": SHIP_CLASS_IDS,
    "aircraft": AIRCRAFT_CLASS_IDS,
    "vehicle": VEHICLE_CLASS_IDS,
}


def class_group(category_id: int) -> str:
    for group, class_ids in GROUP_CLASS_IDS.items():
        if category_id in class_ids:
            return group
    raise ValueError(f"unknown category_id: {category_id}")
