"""Offline end-to-end demo: resume text -> routed skill hits (no API keys).

`--live` tries the real LiteLLM chain from configs/litellm.yaml and falls back
to the offline demo with a notice when keys/server/deps are missing.
"""
from __future__ import annotations

import argparse
import json

from skill_project.gateway import FakeClient, SkillExtractor, SkillRouter
from skill_project.taxonomy import CompositeTaxonomy, CustomSkillRegistry, InMemoryTaxonomy, Skill

RESUME = ("Senior data engineer: 5 years of Python programming and ETL pipelines, "
          "Docker for deploys, plus prompt engineering for internal LLM tools.")

CANNED = json.dumps([
    {"skill_id": "esco:seed-python", "span": "Python programming"},
    {"skill_id": "esco:seed-etl", "span": "ETL pipelines"},
    {"skill_id": "esco:seed-docker", "span": "Docker"},
    {"skill_id": "custom:prompt-engineering", "span": "prompt engineering"},
])


def build_demo_taxonomy() -> CompositeTaxonomy:
    base = InMemoryTaxonomy([
        Skill(id="esco:seed-python", label="Python programming", namespace="esco"),
        Skill(id="esco:seed-etl", label="ETL pipelines", namespace="esco"),
        Skill(id="esco:seed-docker", label="Container orchestration", namespace="esco"),
    ])
    base.namespace = "esco"
    custom = CustomSkillRegistry([
        Skill(id="custom:prompt-engineering", label="prompt engineering", namespace="custom"),
    ])
    return CompositeTaxonomy([base, custom])


def offline_demo() -> int:
    taxonomy = build_demo_taxonomy()
    router = SkillRouter(FakeClient(CANNED), taxonomy,
                         ["skill-detect-qlora", "skill-detect-fallback"])
    hits = router.extract(RESUME)
    print(f"model: {router.used_model}")
    for h in hits:
        print(f"{h.skill_id} | {h.label} | {h.namespace} | {h.span}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args(argv)
    if not args.live:
        return offline_demo()
    try:
        import yaml
        from skill_project.gateway import LiteLLMClient
        with open("configs/litellm.yaml", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        chain = [m["model_name"] for m in cfg["model_list"]]
        router = SkillRouter(LiteLLMClient(), build_demo_taxonomy(), chain)
        hits = router.extract(RESUME)
        print(f"live model: {router.used_model}")
        for h in hits:
            print(f"{h.skill_id} | {h.label} | {h.namespace} | {h.span}")
        return 0
    except Exception as exc:
        print(f"live unavailable ({exc}); offline demo instead:")
        return offline_demo()


if __name__ == "__main__":
    raise SystemExit(main())
