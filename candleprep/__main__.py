import argparse
import json
from .datasets import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate verified CPU candle corruption datasets")
    parser.add_argument("source", help="Original candle CSV (never modified)")
    parser.add_argument("--output", default="generated_data")
    parser.add_argument("--train", type=int, default=10000)
    parser.add_argument("--validation", type=int, default=1000)
    parser.add_argument("--test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=4820)
    args = parser.parse_args()
    manifest = generate_dataset(args.source, args.output, args.train, args.validation, args.test, args.seed)
    print(json.dumps({name: {key: split[key] for key in ("examples", "classes", "categories")}
                      for name, split in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
