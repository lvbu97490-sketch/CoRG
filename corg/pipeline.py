"""CoRG for a batch of queries (Algorithm 1 of the paper)."""
from PIL import Image

from .consensus import consensus
from .interface import BOX_PROMPTS, MIMO_MAX_PIXELS, MIMO_MIN_PIXELS, chat_prompt, family, parse_box, parse_point, prompt_text
from .refine import Refiner

SEED = 123
CONTEXT = 8192
NUM_SAMPLES = 8
TEMPERATURE = 0.7
TAU = 0.5


class CoRG:
    """Training-free grounding with a frozen MLLM (served by vLLM) and a frozen SAM 2.1."""

    def __init__(self, model, model_path, sam_checkpoint, gpu_memory_utilization=0.85):
        from transformers import AutoProcessor, AutoTokenizer
        from vllm import LLM
        self.model = model
        fam = family(model)
        kwargs = dict(model=model_path, dtype="bfloat16", max_model_len=CONTEXT, seed=SEED,
                      gpu_memory_utilization=gpu_memory_utilization, limit_mm_per_prompt={"image": 1, "video": 0})
        if fam == "mimo":
            self.processor = AutoProcessor.from_pretrained(model_path)
            kwargs["mm_processor_kwargs"] = {"min_pixels": MIMO_MIN_PIXELS, "max_pixels": MIMO_MAX_PIXELS}
        elif fam == "qwen":
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            kwargs["trust_remote_code"] = True
        else:
            self.processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            kwargs["trust_remote_code"] = True
        self.llm = LLM(**kwargs)
        self.refiner = Refiner(sam_checkpoint)

    def _params(self, n, temperature, max_tokens):
        from vllm import SamplingParams
        return SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n, seed=SEED,
                              skip_special_tokens=family(self.model) != "mimo")

    def _generate(self, queries, images, prompt_id, params):
        reqs = [{"prompt": chat_prompt(self.processor, self.model, prompt_text(self.model, prompt_id, q["expr"])),
                 "multi_modal_data": {"image": img}} for q, img in zip(queries, images)]
        return [[c.text for c in o.outputs] for o in self.llm.generate(reqs, params, use_tqdm=False)]

    def __call__(self, queries):
        """queries: list of dicts with "image" (path) and "expr". Returns one record per query."""
        images = [Image.open(q["image"]).convert("RGB") for q in queries]
        sizes = [img.size for img in images]
        greedy, point_params = self._params(1, 0.0, 128), self._params(1, 0.0, 64)
        # Interface: six prompts with greedy decoding, eight samples under the default prompt, and the point query
        outs = {pid: self._generate(queries, images, pid, greedy) for pid in BOX_PROMPTS}
        samples = self._generate(queries, images, "P1", self._params(NUM_SAMPLES, TEMPERATURE, 128))
        points = self._generate(queries, images, "Ppt", point_params)
        results = []
        for i, q in enumerate(queries):
            W, H = sizes[i]
            prompt_boxes = {pid: parse_box(self.model, outs[pid][i][0], W, H) for pid in BOX_PROMPTS}
            sample_boxes = [b for b in (parse_box(self.model, t, W, H) for t in samples[i]) if b is not None]
            hypotheses = [b for b in prompt_boxes.values() if b is not None] + sample_boxes
            # Hypothesis consensus
            box = consensus(hypotheses, TAU)
            # Consensus-guided refinement with the geometric guard
            point = parse_point(self.model, points[i][0], W, H)
            if box is not None and point is not None:
                self.refiner.set_image(images[i], key=q["image"])
            final = self.refiner(box, point)
            results.append({**q, "direct": prompt_boxes["P1"], "hypotheses": hypotheses, "consensus": box,
                            "point": point, "box": final})
        return results
