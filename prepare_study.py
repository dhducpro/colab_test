"""Fetch pinned instances for the family comparison and the original pilot checks."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

THAO = "thaonc97/thao_phd_model/c85c247550f6232e6aec519b15824e283ee87883"
PILOT = "dhducpro/PDPCR_Pilot_GA_TSR_MLS/6c70b2da70645f45536ca5fe55169c1ade083b7d"


def prepare(data_dir, mode):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    family = [f"{prefix}{i}-5-uniform.csv" for prefix in ("lc", "lr", "lrc") for i in range(101, 106)]
    pilot = []
    for family_name in ("lc", "lr", "lrc"):
        pilot.extend([
            f"lilim_{family_name}101_n50_m10_req0_first50.json",
            f"lilim_{family_name}1_2_1_n100_m10_req0_first100.json",
            f"lilim_{family_name}1_4_1_n200_m10_req0_first200.json",
        ])
    if mode in {"families", "ablation"}:
        pilot = []
    else:
        family = []
        if mode != "pilot":
            tags = (("lc101_n50", "lrc101_n50", "lr1_2_1_n100") if mode == "calibration"
                    else ("lrc101_n50", "lrc1_2_1_n100", "lrc1_4_1_n200"))
            pilot = [name for name in pilot if any(tag in name for tag in tags)]
    entries = [(f"{THAO}/data/lim/100/{name}", name) for name in family]
    entries += [(f"{PILOT}/instances/{name}", name) for name in pilot]

    def fetch(item):
        url, name = item
        with urlopen("https://raw.githubusercontent.com/" + url, timeout=60) as response:
            data = response.read()
        target = data_dir / name
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(fetch, entries))
    return [data_dir / name for name in family], [data_dir / name for name in pilot]
