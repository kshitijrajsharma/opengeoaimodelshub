import argparse
import json
import zipfile
from pathlib import Path

def stacmlm_to_emd(stac_path: Path, output_dir: Path) -> Path:
    """Convert STAC-MLM item to Esri .emd format."""
    stac = json.loads(stac_path.read_text())
    props = stac["properties"]

    onnx_asset = next((v for v in stac["assets"].values() if v.get("mlm:artifact_type") == "onnx"), None)
    if not onnx_asset:
        raise ValueError("No ONNX model found in STAC MLM assets.")

    input_desc = props["mlm:input"][0]
    output_desc = props["mlm:output"][0]
    bands = input_desc.get("bands", ["red", "green", "blue"])
    class_defs = output_desc.get("classification:classes", [])

    band_indices = [bands.index(b) + 1 for b in ["red", "green", "blue"] if b in bands]

    emd = {
        "Framework": "onnx",
        "ModelType": "SemanticSegmentation",
        "Architecture": props.get("mlm:architecture", "U-Net"),
        "ModelFile": "model.onnx",
        "ImageHeight": input_desc["input"]["shape"][2],
        "ImageWidth": input_desc["input"]["shape"][3],
        "ExtractBands": band_indices,
        "Classes": [{"Value": c["value"], "Name": c["name"]} for c in class_defs],
        "DataRange": [0, 1],
        "BatchSize": 1,
        "ImageType": "RGB"
    }

    emd_path = f"{output_dir}/model.emd"
    with open(emd_path, "w") as f:
        json.dump(emd, f, indent=2)
    return emd_path

def create_dlpk(emd_path: Path, onnx_path: Path, output_dlpk: Path):
    """Package EMD and ONNX model into a .dlpk archive."""
    with zipfile.ZipFile(output_dlpk, 'w') as zipf:
        zipf.write(emd_path, arcname="model.emd")
        zipf.write(onnx_path, arcname="model.onnx")

def main():
    parser = argparse.ArgumentParser(description="Convert STAC-MLM to Esri .dlpk package")
    parser.add_argument("--stac", required=True, help="Path to STAC-MLM JSON")
    parser.add_argument("--onnx", required=False, help="Path to ONNX model (optional; inferred from STAC if relative)")
    parser.add_argument("--out-dir", default="output", help="Output directory (default: ./output)")
    parser.add_argument("--dlpk-name", default="model.dlpk", help="Name for .dlpk file (default: model.dlpk)")
    args = parser.parse_args()

    stac_path = Path(args.stac)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.onnx:
        onnx_path = Path(args.onnx)
    else:
        stac = json.loads(stac_path.read_text())
        onnx_asset = next((v for v in stac["assets"].values() if v.get("mlm:artifact_type") == "onnx"), None)
        if not onnx_asset:
            raise ValueError("No ONNX model asset found in STAC file.")
        onnx_path = stac_path.parent / onnx_asset["href"]

    emd_path = stacmlm_to_emd(stac_path, output_dir)
    dlpk_path = output_dir / args.dlpk_name
    create_dlpk(emd_path, onnx_path, dlpk_path)

    print(f"DLPK created at: {dlpk_path}")

if __name__ == "__main__":
    main()
