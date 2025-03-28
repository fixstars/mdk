import gc
import glob
import json
import logging
import os
import random
import time
from abc import ABC
from collections import defaultdict
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

import deepspeed
import torch
import vllm
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

logger = logging.getLogger(__name__)


class LocalLLM:
    def __init__(self, args):
        preprocess_begin = time.time()
        self.rnd = random.Random(args.seed)
        self.args = args
        self._prepare_model()
        self.examples = []
        for example in args.prompt.with_examples.reference:
            for p in glob.glob(os.path.join(example, "./**")):
                if os.path.isfile(p):
                    self.examples.append(p)
        if len(self.examples) == 0:
            logger.warning("no examples found. converting without examples...")
        jinja_dir = os.path.join(os.path.dirname(__file__), "../../templates")
        jinja_env = Environment(loader=FileSystemLoader(jinja_dir))
        self.template = jinja_env.get_template(args.prompt.template)
        logger.info("init PreprocessWithLLM: %f sec", time.time() - preprocess_begin)

    def _prepare_model(self):
        torch.cuda.set_device(self.args.rank)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.llm.model, padding_side="left"
        )
        # TODO: use deepspeed.OnDevice without corrupting output

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.args.llm.provider == "huggingface":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.llm.model, torch_dtype=torch.float16, device_map="auto"
            )
        elif self.args.llm.provider == "deepspeed":
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.args.llm.model, torch_dtype=torch.float16
            )
            self.model = deepspeed.init_inference(
                hf_model,
                mp_size=self.args.world_size,
                dtype=torch.float16,
                checkpoint=None,
                replace_with_kernel_inject=False,
                max_tokens=self.args.llm.max_tokens,
            )
        else:
            raise ValueError(self.args.llm.provider)

    def _get_examples(self):
        k = self.args.prompt.with_examples.num_sample
        if len(self.examples) == 0 or k == 0:
            return ""

        ret = ""
        for file in self.rnd.choices(self.examples, k=k):
            with open(file, encoding="utf-8") as f:
                ret += f.read()
        return ret

    def _process_model(self, prompts: list[str]) -> list[list[str]]:
        n = self.args.llm.generate.num_return_sequences

        def process_one_batch(batch: list[str]) -> list[str]:
            input_ids = self.tokenizer.batch_encode_plus(
                batch, add_special_tokens=False, return_tensors="pt", padding=True
            )
            tokens = self.model.generate(
                **input_ids.to(device="cuda" if torch.cuda.is_available() else "cpu"),
                **self.args.llm.generate,
            )
            outputs = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
            reply = [outputs[i][len(batch[i // n]) :] for i in range(len(outputs))]
            assert len(reply) == len(batch) * n, (len(reply), len(batch))
            return reply

        dataloader = DataLoader(prompts, batch_size=self.args.llm.batch_size)

        outputs = []
        for batch in tqdm(dataloader, desc="process model"):
            outputs += process_one_batch(batch)

        # [A, A, B, B, C, C] -> [[A, A], [B, B], [C, C]]
        outputs = [outputs[i : i + n] for i in range(0, len(outputs), n)]
        return outputs

    def _combine_prompt(self, pages: list[dict[str, str]]) -> list[str]:
        prompts = []
        for page in pages:
            examples = self._get_examples()
            prompts.append(
                self.template.render(
                    topic=self.args.prompt.topic, examples=examples, **page
                )
            )
        return prompts

    def __call__(self, pages: list[dict[str, str]]):
        prompts = self._combine_prompt(pages)
        process_begin = time.time()
        outputs = self._process_model(prompts)
        process_end = time.time()
        tokenizer = AutoTokenizer.from_pretrained(self.args.llm.model)
        num_tokens = sum([len(tokenizer.encode(s)) for s in sum(outputs, [])])
        logger.info(
            "llm: %f token/sec (%d token / %f sec) with %s",
            num_tokens / (process_end - process_begin),
            num_tokens,
            process_end - process_begin,
            self.args.llm,
        )
        assert len(pages) == len(prompts) == len(outputs), (
            len(pages),
            len(prompts),
            len(outputs),
        )
        assert len(outputs[0]) == self.args.llm.generate.num_return_sequences
        return outputs, prompts

    def empty_cache(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class OpenAICompatibleLLM(LocalLLM):
    def _prepare_model(self):
        self.client = OpenAI(
            base_url=self.args.llm.api_base_url,
            api_key=self.args.llm.api_key,
        )

    def _process_model(self, prompts: list[str]):
        stream = self.client.completions.create(
            model=self.args.llm.model,
            prompt=prompts,
            max_tokens=self.args.llm.generate.max_new_tokens,
            n=self.args.llm.generate.num_return_sequences,
            stop=None,
            temperature=self.args.llm.generate.temperature,
            top_p=self.args.llm.generate.top_p,
            stream=True,
        )
        response = defaultdict(str)
        finished_ids = set()
        n = self.args.llm.generate.num_return_sequences
        progress_bar = tqdm(range(len(prompts) * n))
        for chunk in stream:
            response[chunk.choices[0].index] += chunk.choices[0].text
            if (
                chunk.choices[0].finish_reason is not None
                and chunk.choices[0].index not in finished_ids
            ):
                finished_ids.add(chunk.choices[0].index)
                progress_bar.update()
        outputs = [[response[i * n + j] for j in range(n)] for i in range(len(prompts))]
        return outputs

    def empty_cache(self):
        pass


class Commands:
    INITIALIZE = "initialize"
    GENERATE = "generate"
    UNLOAD = "unload"


class PipeMessageABC(ABC):
    def send(self):
        pass

    def receive(self):
        pass


class InitializeMessage(PipeMessageABC):
    def send(
        self,
        tensor_parallel_size: int,
        max_num_seqs: int,
        model: str,
    ) -> tuple[Connection, Process]:
        parent_conn, child_conn = Pipe()
        parent_conn.send(
            f"{Commands.INITIALIZE}:{tensor_parallel_size}:{max_num_seqs}:{model}"
        )
        process_a = Process(target=vllm_worker, args=(child_conn,))
        process_a.start()
        parent_conn.recv()
        return (parent_conn, process_a)

    def receive(self, conn: Connection, message: str) -> vllm.LLM:
        ms = message.split(":")
        llm = vllm.LLM(
            model="".join(ms[3:]),
            tensor_parallel_size=int(ms[1]),
            max_num_seqs=int(ms[2]),
            distributed_executor_backend="ray",  # for vllm-0.5.x
        )
        conn.send("done")
        return llm


class GenerateMessage(PipeMessageABC):
    def send(
        self,
        parent_conn: Connection,
        prompts: list[str],
        sampling_params: vllm.SamplingParams,
    ) -> dict:
        params = (
            f"{sampling_params.n},{sampling_params.temperature},"
            + f"{sampling_params.top_p},{sampling_params.max_tokens},{sampling_params.seed}"
        )
        json_prompts = json.dumps(prompts)
        message = f"{Commands.GENERATE}:{params}:{json_prompts}"
        parent_conn.send(message)
        raw_response = parent_conn.recv()
        json_response = json.loads(raw_response)
        return json_response

    def receive(self, conn: Connection, message: str, llm: vllm.LLM):
        ms = message.split(":")
        param = ms[1].split(",")
        assert len(param) == 5, "Wrong message format for SamplingParams"
        sampling_params = vllm.SamplingParams(
            n=int(param[0]),
            temperature=float(param[1]),
            top_p=float(param[2]),
            max_tokens=int(param[3]),
            seed=int(param[4]),
        )
        json_prompts = "".join(ms[2:])
        prompts = json.loads(json_prompts)
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        results = []
        for i in range(len(outputs)):
            results.append([o.text for o in outputs[i].outputs])
        json_results = json.dumps(results)
        conn.send(json_results)


class UnloadMessage(PipeMessageABC):
    def send(self, parent_conn: Connection, process_a: Process):
        parent_conn.send(Commands.UNLOAD)
        process_a.join()

    def receive(self, _: Connection, llm: vllm.LLM):
        destroy_model_parallel()
        del llm


def vllm_worker(conn: Connection):
    llm = None

    while True:
        message = conn.recv()
        command = message.split(":")[0]

        if command == Commands.INITIALIZE:
            llm = InitializeMessage().receive(conn, message)
        elif command == Commands.GENERATE:
            GenerateMessage().receive(conn, message, llm)
        elif command == Commands.UNLOAD:
            UnloadMessage().receive(conn, llm)
            break


class VLLM_Generator:
    def __init__(self, model: str, tensor_parallel_size: int, max_num_seqs: int):
        self.parent_conn, self.process_a = InitializeMessage().send(
            tensor_parallel_size, max_num_seqs, model
        )

    def generate(self, prompts: list[str], sampling_params: vllm.SamplingParams):
        json_response = GenerateMessage().send(
            self.parent_conn, prompts, sampling_params
        )
        return json_response

    def unload(self):
        UnloadMessage().send(self.parent_conn, self.process_a)


class VLLM_Multiprocessing(LocalLLM):
    """https://github.com/vllm-project/vllm/issues/1908
    There's a bug in VLLM v0.5.x where memory isn't released after running multiple models consecutively.
    To resolve this, we'll use multiprocessing to run VLLM in a separate process.
    """

    def _prepare_model(self):
        torch.multiprocessing.set_start_method("spawn")
        gpus = self.args.llm.vllm_gpus
        tp = self.args.llm.vllm_tensor_parallel_size
        assert tp == len(gpus.split(",")), "Set tensor_parallel_size == available GPUs."
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        self.llm = VLLM_Generator(
            model=self.args.llm.model,
            tensor_parallel_size=tp,
            max_num_seqs=self.args.llm.vllm_max_num_seqs,
        )
        self.sampling_params = vllm.SamplingParams(
            n=self.args.llm.generate.num_return_sequences,
            temperature=self.args.llm.generate.temperature,
            top_p=self.args.llm.generate.top_p,
            max_tokens=self.args.llm.generate.max_new_tokens,
            seed=self.args.seed,
        )

    def _process_model(self, prompts: list[str]):
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        return outputs

    def empty_cache(self):
        self.llm.unload()


class VLLM(LocalLLM):
    def _prepare_model(self):
        self.llm = vllm.LLM(
            model=self.args.llm.model,
            tensor_parallel_size=self.args.llm.vllm_tensor_parallel_size,
            max_num_seqs=self.args.llm.vllm_max_num_seqs,
        )
        self.sampling_params = vllm.SamplingParams(
            n=self.args.llm.generate.num_return_sequences,
            temperature=self.args.llm.generate.temperature,
            top_p=self.args.llm.generate.top_p,
            max_tokens=self.args.llm.generate.max_new_tokens,
            seed=self.args.seed,
        )

    def _process_model(self, prompts: list[str]):
        response = self.llm.generate(prompts, sampling_params=self.sampling_params)
        outputs = [
            [
                response[i].outputs[j].text
                for j in range(self.args.llm.generate.num_return_sequences)
            ]
            for i in range(len(prompts))
        ]
        return outputs

    def empty_cache(self):
        # cf. https://github.com/vllm-project/vllm/issues/1908#issuecomment-2094146933
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.llm.llm_engine.model_executor.driver_worker
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()


def load_llm(args):
    if args.llm.provider in ["huggingface", "deepspeed"]:
        return LocalLLM(args)
    elif args.llm.provider == "api":
        return OpenAICompatibleLLM(args)
    elif args.llm.provider == "vllm_multiprocessing":
        return VLLM_Multiprocessing(args)
    elif args.llm.provider == "vllm":
        return VLLM(args)
    else:
        raise ValueError(args.llm.provider)
