import argparse
import json
import zipfile
from pathlib import Path


def stacmlm_to_emd(stac_path: Path, output_dir: Path) -> Path:
    stac = json.loads(stac_path.read_text())
    props = stac["properties"]

    pt_asset = next(
        (v for v in stac["assets"].values() if v.get("mlm:artifact_type") == "pytorch"),
        None,
    )
    if not pt_asset:
        raise ValueError("No PyTorch model found in STAC MLM assets.")

    input_desc = props["mlm:input"][0]
    output_desc = props["mlm:output"][0]
    bands = input_desc.get("bands", ["red", "green", "blue"])
    class_defs = output_desc.get("classification:classes", [])

    emd = {
        "Framework": "PyTorch",
        "ModelType": "ImageClassification",
        "ModelName": props.get("mlm:name", "RefugeeCampDetector"),
        "Architecture": props.get("mlm:architecture", "U-Net"),
        "ModelFile": "model.pt",
        "InferenceFunction": "RefugeeCampDetector.py",
        "ImageHeight": input_desc["input"]["shape"][2],
        "ImageWidth": input_desc["input"]["shape"][3],
        "ImageSpaceUsed": "MAP_SPACE",
        "ExtractBands": [0, 1, 2],
        "DataRange": [0, 1],
        "BatchSize": 1,
        "ImageType": "RGB",
        "Classes": [
            {"Value": 0, "Name": "Background", "Color": [0, 0, 0]},
            {"Value": 1, "Name": "Refugee Camp", "Color": [255, 0, 0]},
        ],
        "ModelParameters": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "Threshold": 0.5,
    }

    emd_path = output_dir / "model.emd"
    with emd_path.open("w") as f:
        json.dump(emd, f, indent=2)
    return emd_path


def create_dlpk(emd_path: Path, pt_path: Path, inference_path: Path, output_dlpk: Path):
    with zipfile.ZipFile(output_dlpk, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(emd_path, arcname="model.emd")
        zipf.write(pt_path, arcname="model.pt")
        zipf.write(inference_path, arcname="RefugeeCampDetector.py")


def main():
    parser = argparse.ArgumentParser(
        description="Convert STAC-MLM to Esri .dlpk package"
    )
    parser.add_argument("--stac", required=True, help="Path to STAC-MLM JSON")
    parser.add_argument(
        "--pt",
        required=False,
        help="Path to PyTorch model (optional; inferred from STAC if relative)",
    )
    parser.add_argument(
        "--out-dir", default="output", help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--dlpk-name",
        default="model.dlpk",
        help="Name for .dlpk file (default: model.dlpk)",
    )
    args = parser.parse_args()

    stac_path = Path(args.stac)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pt:
        pt_path = Path(args.pt)
    else:
        stac = json.loads(stac_path.read_text())
        pt_asset = next(
            (
                v
                for v in stac["assets"].values()
                if v.get("mlm:artifact_type") == "pytorch"
            ),
            None,
        )
        if not pt_asset:
            raise ValueError("No PyTorch model asset found in STAC file.")
        pt_path = (stac_path.parent / pt_asset["href"]).resolve()

    # Look for inference.py in the same directory as the script
    inference_path = Path(__file__).parent / "inference.py"
    if not inference_path.exists():
        raise ValueError(f"inference.py not found at {inference_path}")

    emd_path = stacmlm_to_emd(stac_path, output_dir)
    dlpk_path = output_dir / args.dlpk_name
    create_dlpk(emd_path, pt_path, inference_path, dlpk_path)

    print(f"DLPK created at: {dlpk_path}")


if __name__ == "__main__":
    main()
