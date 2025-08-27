#!/usr/bin/env python3

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

import jsonschema
import requests


def load_stac_schema(version: str = "v1.5.0") -> Dict[str, Any]:
    schema_url = f"https://stac-extensions.github.io/mlm/{version}/schema.json"
    try:
        response = requests.get(schema_url, timeout=30)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, requests.exceptions.Timeout) as e:
        print(f"Error loading schema from {schema_url}: {e}")
        sys.exit(1)


def validate_stac_item(item: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    errors = []
    try:
        jsonschema.validate(item, schema)
        print("✅ STAC-MLM schema validation passed!")
    except jsonschema.exceptions.ValidationError as e:
        errors.append(f"Schema validation error: {e.message}")
        if e.path:
            errors.append(f"  Path: {' -> '.join(str(p) for p in e.path)}")
    except (jsonschema.exceptions.SchemaError, ValueError) as e:
        errors.append(f"Validation error: {str(e)}")
    
    return errors


def check_completeness(item: Dict[str, Any]) -> Dict[str, Any]:
    properties = item.get("properties", {})
    assets = item.get("assets", {})
    
    required_fields = ["mlm:name", "mlm:architecture", "mlm:tasks", "mlm:framework"]
    recommended_fields = [
        "mlm:framework_version", "mlm:memory_size", "mlm:total_parameters",
        "mlm:pretrained", "mlm:accelerator", "mlm:batch_size_suggestion"
    ]
    
    required_present = sum(1 for field in required_fields if field in properties)
    recommended_present = sum(1 for field in recommended_fields if field in properties)
    
    model_assets = sum(1 for asset in assets.values() if "mlm:model" in asset.get("roles", []))
    
    inputs = properties.get("mlm:input", [])
    outputs = properties.get("mlm:output", [])
    inputs_with_scaling = sum(1 for inp in inputs if "value_scaling" in inp)
    outputs_with_classes = sum(1 for out in outputs if "classification:classes" in out)
    
    return {
        "required_fields": f"{required_present}/{len(required_fields)}",
        "recommended_fields": f"{recommended_present}/{len(recommended_fields)}",
        "model_assets": model_assets,
        "total_assets": len(assets),
        "inputs_with_scaling": f"{inputs_with_scaling}/{len(inputs)}" if inputs else "0/0",
        "outputs_with_classes": f"{outputs_with_classes}/{len(outputs)}" if outputs else "0/0"
    }


def calculate_quality_score(item: Dict[str, Any]) -> float:
    properties = item.get("properties", {})
    assets = item.get("assets", {})
    
    score = 0.0
    
    required_fields = ["mlm:name", "mlm:architecture", "mlm:tasks", "mlm:framework"]
    score += sum(20 for field in required_fields if field in properties)
    
    recommended_fields = [
        "mlm:framework_version", "mlm:memory_size", "mlm:total_parameters",
        "mlm:pretrained", "mlm:accelerator", "mlm:batch_size_suggestion"
    ]
    score += sum(5 for field in recommended_fields if field in properties)
    
    if any("mlm:model" in asset.get("roles", []) for asset in assets.values()):
        score += 10
    
    return min(score, 100.0)


def main():
    parser = argparse.ArgumentParser(description="Validate STAC-MLM items")
    parser.add_argument("stac_file", help="Path to STAC-MLM JSON file")
    parser.add_argument("--schema-version", default="v1.5.0", help="STAC-MLM schema version")
    
    args = parser.parse_args()
    
    stac_file = Path(args.stac_file)
    if not stac_file.exists():
        print(f"Error: File {stac_file} not found")
        sys.exit(1)
    
    with open(stac_file, 'r') as f:
        item = json.load(f)
    
    print(f"Validating STAC-MLM item: {stac_file}")
    print(f"Item ID: {item.get('id', 'Unknown')}")
    print(f"Model: {item.get('properties', {}).get('mlm:name', 'Unknown')}")
    print("-" * 50)
    
    schema = load_stac_schema(args.schema_version)
    errors = validate_stac_item(item, schema)
    
    if errors:
        print("❌ VALIDATION ERRORS:")
        for error in errors:
            print(f"  {error}")
        print()
    
    completeness = check_completeness(item)
    quality_score = calculate_quality_score(item)
    
    print(f"📊 QUALITY SCORE: {quality_score}/100")
    print()
    print("📈 COMPLETENESS:")
    print(f"  • Required Fields: {completeness['required_fields']}")
    print(f"  • Recommended Fields: {completeness['recommended_fields']}")
    print(f"  • Model Assets: {completeness['model_assets']}")
    print(f"  • Total Assets: {completeness['total_assets']}")
    print(f"  • Inputs With Scaling: {completeness['inputs_with_scaling']}")
    print(f"  • Outputs With Classes: {completeness['outputs_with_classes']}")
    
    if not errors and quality_score == 100.0:
        print()
        print("✅ STAC-MLM item is production-ready!")
    
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
