"""Source registry and repo cloner. Pins = SHA recorded at acquisition."""
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceSpec:
    name: str
    git_url: str
    tier: str          # "train" | "eval"
    license_note: str
    world_glob: str    # glob under the repo root for world files
    parser: str        # "wld" | "are" | "coffeemud"


SOURCES: dict[str, SourceSpec] = {
    "tbamud": SourceSpec("tbamud", "https://github.com/tbamud/tbamud.git",
                         "train", "CircleMUD/DikuMUD, LGPL since 2020",
                         "lib/world/wld/*.wld", "wld"),
    "coffeemud": SourceSpec("coffeemud",
                            "https://github.com/bozimmerman/CoffeeMud.git",
                            "train", "Apache-2.0",
                            "resources/**/*.cmare", "coffeemud"),
    "smaug": SourceSpec("smaug", "https://github.com/smaugmuds/_smaug_.git",
                        "eval", "Diku/Merc/SMAUG non-commercial chain",
                        "**/area/*.are", "are"),
    "rom": SourceSpec("rom", "https://github.com/avinson/rom24-quickmud.git",
                      "eval", "Diku/Merc/ROM non-commercial chain",
                      "area/*.are", "are"),
    "awakemud": SourceSpec("awakemud",
                           "https://github.com/luciensadi/AwakeMUD.git",
                           "eval", "Circle lineage + Shadowrun fan IP",
                           "lib/world/wld/*.wld", "wld"),
    "swfote": SourceSpec("swfote", "https://github.com/Arthmoor/SWFOTEFUSS.git",
                         "eval", "SMAUG lineage + Star Wars fan IP",
                         "area/*.are", "are"),
}


def clone_all(dest: Path, only: list[str] | None = None) -> dict[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    pins: dict[str, str] = {}
    if (dest / "PINS.json").exists():
        pins = json.loads((dest / "PINS.json").read_text())
    for name, spec in SOURCES.items():
        if only and name not in only:
            continue
        repo = dest / name
        if not repo.exists():
            subprocess.run(["git", "clone", "--depth", "1", spec.git_url,
                            str(repo)], check=True)
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True,
                             check=True).stdout.strip()
        pins[name] = sha
    (dest / "PINS.json").write_text(json.dumps(pins, indent=2))
    return {k: v for k, v in pins.items() if not only or k in only}


if __name__ == "__main__":
    import sys
    result = clone_all(Path("data/raw"),
                       only=sys.argv[1:] or None)
    print(json.dumps(result, indent=2))
