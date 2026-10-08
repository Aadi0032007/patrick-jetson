"""Inspect GGUF metadata without loading tensor weights or installing packages."""
import argparse
import json
from pathlib import Path
import struct


def metadata(path):
    with Path(path).open("rb") as stream:
        def read(fmt):
            size = struct.calcsize("<" + fmt)
            return struct.unpack("<" + fmt, stream.read(size))[0]

        def string():
            size = read("Q")
            if size > 16_000_000:
                raise ValueError("Unexpected GGUF metadata string size")
            return stream.read(size).decode("utf-8")

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                element_type, length = read("I"), read("Q")
                if length > 1_000_000:
                    raise ValueError("Unexpected GGUF metadata array length")
                return [value(element_type) for _ in range(length)]
            return read({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                         6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[kind])

        if stream.read(4) != b"GGUF":
            raise ValueError("Invalid GGUF header")
        version, tensors, count = read("I"), read("Q"), read("Q")
        if version not in (2, 3) or count > 100_000:
            raise ValueError("Unsupported GGUF header")
        result = {"gguf_version": version, "tensor_count": tensors}
        for _ in range(count):
            key = string()
            result[key] = value(read("I"))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    for key, item in metadata(args.path).items():
        if isinstance(item, str) and len(item) > 200:
            item = item[:160] + f"... ({len(item)} characters)"
        elif isinstance(item, list) and len(item) > 12:
            item = f"array with {len(item)} elements"
        print(key + ": " + json.dumps(item, ensure_ascii=True))


if __name__ == "__main__":
    main()
