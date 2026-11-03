"""Run CoRG on one query or on a JSON Lines file of queries.

One query:
  python scripts/run_corg.py --model qwen2.5-vl --model-path Qwen/Qwen2.5-VL-7B-Instruct \
      --sam-checkpoint checkpoints/sam2.1_hiera_large.pt --image demo.jpg --expr "the white car on the left"

Many queries (one JSON object per line with "image" and "expr", other fields are kept):
  python scripts/run_corg.py --model internvl3.5 --model-path OpenGVLab/InternVL3_5-8B \
      --sam-checkpoint checkpoints/sam2.1_hiera_large.pt --queries queries.jsonl --out results.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from corg.interface import MODELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODELS), help="model family of the checkpoint")
    ap.add_argument("--model-path", required=True, help="local path or Hugging Face name of the MLLM")
    ap.add_argument("--sam-checkpoint", required=True, help="SAM 2.1 Hiera-L checkpoint")
    ap.add_argument("--image")
    ap.add_argument("--expr")
    ap.add_argument("--queries", help="JSON Lines file of queries")
    ap.add_argument("--out", help="output JSON Lines file (default: print)")
    ap.add_argument("--batch", type=int, default=512, help="queries per vLLM batch")
    a = ap.parse_args()
    if a.queries:
        queries = [json.loads(line) for line in open(a.queries, encoding="utf-8") if line.strip()]
    elif a.image and a.expr:
        queries = [{"image": a.image, "expr": a.expr}]
    else:
        ap.error("give --queries, or --image and --expr")

    from corg.pipeline import CoRG
    corg = CoRG(a.model, a.model_path, a.sam_checkpoint)
    out = open(a.out, "w", encoding="utf-8") if a.out else sys.stdout
    for s in range(0, len(queries), a.batch):
        for r in corg(queries[s:s + a.batch]):
            out.write(json.dumps(r) + "\n")
        out.flush()
    if a.out:
        out.close()


if __name__ == "__main__":
    main()
