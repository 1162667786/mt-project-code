import argparse
import json
from comet import load_from_checkpoint


def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def to_jsonable(obj):
    """
    Recursively convert COMET / xCOMET outputs to JSON-serializable objects.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]

    # numpy / torch scalar
    try:
        return obj.item()
    except Exception:
        pass

    # dataclass / namedtuple / custom object
    if hasattr(obj, "__dict__"):
        return {str(k): to_jsonable(v) for k, v in vars(obj).items()}

    return str(obj)


def get_per_sample_metadata(metadata, i, n):
    """
    xCOMET metadata may appear as:
    1. list[dict], one dict per sample
    2. dict[str, list], each key has n elements
    3. custom object
    This function tries to extract sample-level metadata robustly.
    """
    if metadata is None:
        return None

    metadata_json = to_jsonable(metadata)

    # Case 1: list with one metadata item per sample
    if isinstance(metadata_json, list):
        if len(metadata_json) == n:
            return metadata_json[i]
        return metadata_json

    # Case 2: dict where values are per-sample lists
    if isinstance(metadata_json, dict):
        sample_meta = {}
        extracted_any = False

        for k, v in metadata_json.items():
            if isinstance(v, list) and len(v) == n:
                sample_meta[k] = v[i]
                extracted_any = True
            else:
                sample_meta[k] = v

        if extracted_any:
            return sample_meta

        return metadata_json

    return metadata_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--xcomet_model", required=True)
    parser.add_argument("--src_field", default="src")
    parser.add_argument("--mt_field", default="tgt")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--save_raw_attrs", action="store_true")
    args = parser.parse_args()

    data = read_jsonl(args.input_file)

    comet_data = []
    for item in data:
        comet_data.append({
            "src": item[args.src_field],
            "mt": item[args.mt_field],
        })

    print(f"Loaded {len(comet_data)} samples.")
    print(f"src_field = {args.src_field}")
    print(f"mt_field  = {args.mt_field}")
    print(f"Loading xCOMET model from: {args.xcomet_model}")

    model = load_from_checkpoint(args.xcomet_model)

    print("Running xCOMET prediction...")
    output = model.predict(
        comet_data,
        batch_size=args.batch_size,
        gpus=args.gpus,
    )

    # Basic score output
    scores = output.scores

    # xCOMET explanation metadata, usually contains error spans / severity / confidence
    metadata = getattr(output, "metadata", None)

    print("Prediction output type:", type(output))
    print("Available output attributes:", dir(output))

    if metadata is None:
        print(
            "WARNING: output.metadata is None. "
            "This xCOMET checkpoint or COMET version may not be returning error spans."
        )
    else:
        print("Metadata type:", type(metadata))

    n = len(data)

    for i, (item, score) in enumerate(zip(data, scores)):
        item["xcomet_score"] = float(score)

        sample_meta = get_per_sample_metadata(metadata, i, n)
        if sample_meta is not None:
            item["xcomet_metadata"] = sample_meta

            # Try to expose common xCOMET fields at top level for easier analysis
            if isinstance(sample_meta, dict):
                for key in [
                    "error_spans",
                    "errors",
                    "minor_errors",
                    "major_errors",
                    "critical_errors",
                    "mqm_score",
                    "confidence",
                    "severity",
                ]:
                    if key in sample_meta:
                        item[f"xcomet_{key}"] = sample_meta[key]

        if args.save_raw_attrs:
            # This is useful for debugging but may make the output large.
            item["xcomet_raw_output_attrs"] = {
                k: to_jsonable(getattr(output, k))
                for k in dir(output)
                if not k.startswith("_") and k not in ["scores", "metadata"]
            }

    write_jsonl(args.output_file, data)

    avg_score = sum(scores) / len(scores)
    print(f"Saved to: {args.output_file}")
    print(f"Average xCOMET score: {avg_score:.4f}")
    print(f"Min xCOMET score: {min(scores):.4f}")
    print(f"Max xCOMET score: {max(scores):.4f}")


if __name__ == "__main__":
    main()