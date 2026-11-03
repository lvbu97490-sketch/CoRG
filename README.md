# CoRG: Unlocking the Localization Capability of Frozen MLLMs for Training-Free Remote Sensing Visual Grounding

CoRG localizes the object described by a natural-language expression in a remote sensing image without any training.
It queries a frozen multimodal large language model (MLLM) through an adaptive grounding interface and replaces its
single prediction with a consensus-then-refinement process.

## Pipeline

For each image and expression, CoRG runs four steps (Algorithm 1 of the paper):

1. **Adaptive grounding interface** (`corg/interface.py`). The frozen MLLM is queried with six prompts under greedy
   decoding and sampled eight times under the default prompt at temperature 0.7, which gives up to 14 box hypotheses.
   A separate point prompt gives the target point. The prompts and the coordinate convention follow the model family,
   and every response is mapped to the pixels of the original image.
2. **Hypothesis consensus** (`corg/consensus.py`). The medoid is the hypothesis with the highest mean IoU to the
   others, and the consensus box averages the hypotheses whose IoU with the medoid is at least 0.5.
3. **Consensus-guided refinement** (`corg/refine.py`). SAM 2.1 receives the consensus box and the target point and
   returns the box of its top-scoring mask.
4. **Geometric guard** (`corg/refine.py`). The refined box is accepted only if the target point lies inside the
   consensus box and the two boxes overlap. Otherwise the consensus box is kept.

`corg/pipeline.py` runs the four steps for a batch of queries, and `scripts/run_corg.py` is the command-line entry.

## Installation

```bash
conda create -n corg python=3.10 -y
conda activate corg
pip install -r requirements.txt
```

Download the SAM 2.1 Hiera-L checkpoint from the
[SAM 2 repository](https://github.com/facebookresearch/sam2) (`sam2.1_hiera_large.pt`).

## Supported MLLMs

| `--model` | Checkpoint | Revision |
|---|---|---|
| `qwen2.5-vl` | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f` |
| `qwen3-vl` | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed8` |
| `qwen3-vl` | `Qwen/Qwen3-VL-32B-Instruct` | `0cfaf48183f5` |
| `internvl3.5` | `OpenGVLab/InternVL3_5-8B` | `9bb6a56ad9cc` |
| `internvl3.5` | `OpenGVLab/InternVL3_5-38B` | `de99855be364` |
| `mimo-vl` | `XiaomiMiMo/MiMo-VL-7B-RL-2508` | `4bfb27076582` |

`--model` selects the prompts and the coordinate convention of the family, and `--model-path` points to the checkpoint.

## Usage

One query:

```bash
python scripts/run_corg.py --model qwen2.5-vl --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --sam-checkpoint checkpoints/sam2.1_hiera_large.pt \
    --image demo.jpg --expr "the white car on the left of the road"
```

A file of queries, one JSON object per line:

```json
{"qid": "q1", "image": "images/00001.jpg", "expr": "the storage tank on the upper right"}
```

```bash
python scripts/run_corg.py --model internvl3.5 --model-path OpenGVLab/InternVL3_5-8B \
    --sam-checkpoint checkpoints/sam2.1_hiera_large.pt \
    --queries queries.jsonl --out results.jsonl
```

Each output line keeps the input fields and adds, in pixels of the original image (x1, y1, x2, y2):

| Field | Content |
|---|---|
| `direct` | box of the default prompt with greedy decoding (the direct prediction) |
| `hypotheses` | valid box hypotheses |
| `consensus` | consensus box |
| `point` | target point (x, y) |
| `box` | final box of CoRG |

From Python:

```python
from corg import CoRG

corg = CoRG("qwen3-vl", "Qwen/Qwen3-VL-8B-Instruct", "checkpoints/sam2.1_hiera_large.pt")
print(corg([{"image": "demo.jpg", "expr": "the tennis court in the middle"}])[0]["box"])
```

## Settings

All backbones share the same settings: six prompts, eight samples at temperature 0.7, agreement threshold 0.5, at most
128 new tokens (64 for the point), a context length of 8192, and seed 123. The MLLM is served with vLLM in bfloat16,
and SAM 2.1 uses multimask output.

## License

This project is released under the [MIT License](LICENSE).
