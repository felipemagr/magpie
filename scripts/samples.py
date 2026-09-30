"""Download a random sample of everyday photos from COCO into samples/."""

import io
import json
import random
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

BASE = "http://images.cocodataset.org"
INDEX = "annotations/image_info_test2017"
OUT = Path("samples")
CONFIG = """[[category]]
name = "pets"
description = "a dog, a cat or another pet"

[[category]]
name = "food"
description = "food or a meal"

[[category]]
name = "vehicles"
description = "a car, bus, train, plane or boat"

[[category]]
name = "sports"
description = "people playing a sport"
"""


def fetch(name: str) -> None:
    (OUT / name).write_bytes(urlopen(f"{BASE}/test2017/{name}").read())


def main(n: int = 200, seed: int = 0) -> None:
    with zipfile.ZipFile(io.BytesIO(urlopen(f"{BASE}/{INDEX}.zip").read())) as z:
        names = [image["file_name"] for image in json.loads(z.read(f"{INDEX}.json"))["images"]]
    OUT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(fetch, random.Random(seed).sample(names, n)))
    if not (config := OUT / "magpie.toml").exists():
        config.write_text(CONFIG)
    print(f"{n} images in {OUT}/")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
