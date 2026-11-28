"""Call the package on one or more images and write JSON lines."""
import argparse
import json
import os

from predict import initialize, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("image", nargs="+")
    args = parser.parse_args()
    os.environ["FASHION_ATTR_DEVICE"] = args.device
    initialize()
    for path in args.image:
        print(json.dumps({"image": path, "attributes": predict(path)},
                         ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()