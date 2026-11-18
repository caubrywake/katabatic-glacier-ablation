"""01 - Validate inputs required by the final manuscript analysis pipeline."""
from __future__ import annotations
from pathlib import Path

from project_config import PROJECT_ROOT
from project_config import RAW_ROOT, PEYTO_RAW, ATHA_RAW

PEYTO_REQUIRED = [
    "ProfileA_tue_20080721_20080728.xlsx",
    "ProfileA_tue_20080803_20080808.xlsx",
    "ProfileA_tue_20080812_20080817.xlsx",
    "ProfileA_tue_20080823_20080825.xlsx",
    "EC_Munro2008_QH_QE.xlsx",
    "EC_Munro2008_QE_corrected20220412.xlsx",
    "PeytoMNH_2008_MOH.obs",
    "Peyto_Qn_hourly.csv",
    "Peyto_Qp_hourly.csv",
    "PeytoSR50_hourly.csv",
    "Peyto_ProfileB_original_consolidated.csv",  # Figure 2
]

ATHA_REQUIRED = [
    "AthaIceProfile.txt",
    "LE_H_atha2015.xlsx",
    "Athabasca_AWSice_hourly.csv",
    "AthabascaSR50_hourly.csv",
    "Athabasca_Qn_hourly.csv",
    "Athabasca_Qp_hourly.csv",
]
ATHA_OPTIONAL = ["iceAWs_Conway.obs"]  # Alternative to the required AWS CSV

def find_one(root: Path, filename: str):
    hits = list(root.rglob(filename))
    return hits[0] if hits else None

def report_group(label: str, root: Path, required: list[str], optional: list[str] | None = None):
    print(f"\n{label}\n{'-' * len(label)}")
    print(f"Searching: {root}")
    if not root.exists():
        print("MISSING FOLDER")
        return {}, required.copy()
    found, missing = {}, []
    for name in required:
        hit = find_one(root, name)
        found[name] = hit
        if hit:
            print(f"OK      {name}\n        {hit}")
        else:
            print(f"MISSING {name}")
            missing.append(name)
    if optional:
        print("\nOptional files")
        for name in optional:
            hit = find_one(root, name)
            found[name] = hit
            if hit:
                print(f"OK      {name}\n        {hit}")
            else:
                print(f"NOT FOUND {name}")
    return found, missing

def check_inputs():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw-data root: {RAW_ROOT}")
    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f"Project root does not exist: {PROJECT_ROOT}")
    peyto_found, peyto_missing = report_group("PEYTO", PEYTO_RAW, PEYTO_REQUIRED)
    atha_found, atha_missing = report_group("ATHABASCA", ATHA_RAW, ATHA_REQUIRED, ATHA_OPTIONAL)
    print("\nSUMMARY\n-------")
    print("Peyto baseline inputs: " + ("COMPLETE" if not peyto_missing else f"MISSING {len(peyto_missing)} file(s)"))
    print("Athabasca minimum inputs: " + ("COMPLETE" if not atha_missing else f"MISSING {len(atha_missing)} file(s)"))
    return {
        "peyto": peyto_found,
        "athabasca": atha_found,
        "peyto_missing": peyto_missing,
        "athabasca_missing": atha_missing,
    }

def check_release_inputs():
    report = check_inputs()
    print("\nAdditional final-pipeline checks:")
    extra = [
        Path(__file__).resolve().parent / "src" / "io_peyto.py",
        Path(__file__).resolve().parent / "src" / "turbulent_fluxes.py",
        Path(__file__).resolve().parent / "src" / "thermodynamics.py",
        Path(__file__).resolve().parent / "src" / "metrics.py",
    ]
    for path in extra:
        print(f"  {'OK' if path.exists() else 'MISSING':7s} {path}")
    print("\nIf all manuscript inputs are OK, run 02_calibrate_katabatic.py next.")
    return report

if __name__ == "__main__":
    result = check_release_inputs()
    if result['peyto_missing'] or result['athabasca_missing']:
        raise SystemExit("Missing required raw inputs; see data/README.md")
