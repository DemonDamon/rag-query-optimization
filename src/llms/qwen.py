# Date    : 2024/7/8 22:59
# File    : Qwen2.py
# Desc    : 参考 https://python.langchain.com/v0.2/docs/how_to/custom_llm/
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com


from abc import ABC
import json
import random
import asyncio
import requests
import numpy as np
from loguru import logger
from threading import Thread
from typing import Any, List, Dict, Mapping, Optional, Iterator, AsyncIterator

import torch
from transformers import TextStreamer
from transformers import TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

from vllm import LLM as VLLM
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
# from minference import MInference
from langchain_core.outputs import GenerationChunk
from langchain.callbacks.manager import CallbackManagerForLLMRun

from base.langchain.llms import LLM


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_tokens):
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids, scores, **kwargs):
        for stop_token in self.stop_tokens:
            if stop_token in input_ids[0][-len(stop_token):]:
                return True
        return False


class Qwen2Local(LLM):
    max_new_token: int = 512
    temperature: float = 0.01  # 修改默认值为0.01，确保为正数
    top_p: float = 1.0  # 0.99
    top_k: int = -1
    tokenizer: object = None
    model: object = None
    device: str = 'cpu'
    streamer: object = None
    use_vllm: bool = False
    sampling_params: object = None
    vllm_gpu_mem_util: float = 0.9
    lora_path: str = None
    request_counter: int = 0
    seed: int = 42

    def __init__(self, model_name_or_path: str, lora_path: str = None, use_vllm: bool = False, vllm_gpu_mem_util: float = 0.9):
        super().__init__()
        self.lora_path = lora_path
        self.use_vllm = use_vllm
        self.vllm_gpu_mem_util = vllm_gpu_mem_util

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 设置随机种子
        set_seed(42)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

        if not self.use_vllm:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map=self.device
            ).eval()

        else:
            self.model = VLLM(model=model_name_or_path,
                              gpu_memory_utilization=self.vllm_gpu_mem_util,
                              dtype="half",  # 适合fp16；如果钥匙胚bf16，这里要改成"bfloat16"
                              max_model_len=106624,  # 根据错误提示设置为可用的最大模型长度
                              enable_lora=True if self.lora_path else False, enable_prefix_caching=True)  # fp16

            # 根据vllm官方文档设置SamplingParams参数
            # 参考: https://docs.vllm.ai/en/latest/dev/sampling_params.html
            self.sampling_params = SamplingParams(
                temperature=self.temperature,  # 控制随机性，0表示贪婪采样
                top_p=self.top_p,  # 核采样概率
                top_k=self.top_k if self.top_k > 0 else -1,  # top-k采样，-1表示考虑所有token
                repetition_penalty=1.0,  # 重复惩罚，1.0表示不惩罚
                max_tokens=self.max_new_token,  # 最大生成token数
                seed=self.seed,  # 随机种子，确保可重现性
                stop=None,  # 停止词列表
                ignore_eos=False  # 是否忽略EOS token
            )
            # minference_patch = MInference("vllm", "Qwen/Qwen2-7B-Instruct")
            # self.model = minference_patch(self.model)

        # self.streamer = TextStreamer(
        #     self.tokenizer, skip_prompt=True, skip_special_tokens=True
        # )

        # 除了使用 TextStreamer 之外，还可以使用 TextIteratorStreamer，
        # 它将可打印的文本存储在一个队列中，以便下游应用程序作为迭代器来使用
        self.streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        self.request_counter = 0

    @property
    def _llm_type(self) -> str:
        return "Qwen2"

    # def save_token_mapping(self, text: str, save_path: str = "token_mapping.txt") -> None:
    #     """保存文本到token的映射关系到文件
        
    #     Args:
    #         text: 要分析的中文文本
    #         save_path: 保存文件的路径
    #     """
    #     # 获取token ids
    #     token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        
    #     # 对每个字符进行分词，获取对应的token id
    #     with open(save_path, "a", encoding="utf-8") as f:
    #         f.write(f"\n{'='*50}\n")
    #         f.write(f"Original text: {text}\n")
    #         f.write(f"Total tokens: {len(token_ids)}\n\n")
            
    #         # 逐字符分析
    #         for char in text:
    #             if char.strip():  # 跳过空白字符
    #                 char_token_ids = self.tokenizer.encode(char, add_special_tokens=False)
    #                 token_text = self.tokenizer.decode(char_token_ids)
    #                 f.write(f"字符: {char} -> Token IDs: {char_token_ids} -> 解码后文本: {token_text}\n")
            
    #         # 完整文本的token信息
    #         f.write(f"\n完整文本的Token IDs: {token_ids}\n")
    #         f.write(f"解码后的完整文本: {self.tokenizer.decode(token_ids)}\n")
    #         f.write(f"{'='*50}\n")

    def save_token_mapping(self, text: str, save_path: str = "token_mapping.txt") -> None:
        """保存文本到token的映射关系到文件，包括空格和换行符等特殊字符
        
        Args:
            text: 要分析的文本
            save_path: 保存文件的路径
        """
        with open(save_path, "a", encoding="utf-8") as f:            
            # 获取完整文本的token ids
            token_ids = self.tokenizer.encode(text)
            f.write(f"Total text len: {len(text)}\n\n")
            f.write(f"Total tokens: {len(token_ids)}\n\n")
            
            # 逐字符分析，包括特殊字符
            current_pos = 0
            while current_pos < len(text):
                # 尝试不同长度的子串，找到最匹配的token
                found_match = False
                for length in range(min(10, len(text) - current_pos), 0, -1):
                    substr = text[current_pos:current_pos + length]
                    substr_token_ids = self.tokenizer.encode(substr, add_special_tokens=False)
                    
                    # 检查这个子串是否能被正确编码和解码
                    decoded_text = self.tokenizer.decode(substr_token_ids)
                    if decoded_text == substr:
                        special_char_repr = repr(substr)[1:-1] if any(c in substr for c in [' ', '\n', '\t', '\r']) else substr
                        f.write(f"文本: {special_char_repr} -> Token IDs: {substr_token_ids} -> Tokens: {decoded_text}\n")
                        current_pos += length
                        found_match = True
                        break
                
                if not found_match:
                    # 如果没找到匹配，则单独处理当前字符
                    char = text[current_pos]
                    char_token_ids = self.tokenizer.encode(char, add_special_tokens=False)
                    char_tokens = self.tokenizer.decode(char_token_ids)
                    special_char_repr = repr(char)[1:-1] if char in [' ', '\n', '\t', '\r'] else char
                    f.write(f"文本: {special_char_repr} -> Token IDs: {char_token_ids} -> Tokens: {char_tokens}\n")
                    current_pos += 1
            
            # 完整文本的token信息
            # f.write(f"\n完整文本的Token IDs: {token_ids}\n")
            # f.write(f"完整文本的Tokens: {tokens}\n")
            # f.write(f"解码后的完整文本: {repr(self.tokenizer.decode(token_ids))}\n")
            # f.write(f"{'='*50}\n")

    def _call(
            self,
            prompt: str,
            history: List[Dict] = [],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            max_new_tokens: Optional[int] = None,
            **kwargs: Any,
    ) -> str:
        self.request_counter += 1
        
        # 使用传入的参数覆盖默认值（如果有提供）
        actual_temperature = temperature if temperature is not None else self.temperature
        actual_top_p = top_p if top_p is not None else self.top_p
        actual_max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_token
        
        # 确保温度参数是严格正数
        if actual_temperature <= 0:
            logger.warning(f"Temperature must be positive, got {actual_temperature}. Using 0.01 instead.")
            actual_temperature = 0.01
        
        # 记录使用的参数
        logger.debug(f"Using parameters: temperature={actual_temperature}, top_p={actual_top_p}, max_new_tokens={actual_max_new_tokens}")
        
        # 定期重置模型缓存
        if self.request_counter % 100 == 0:  # 每100次请求
            # if self.use_vllm:
            #     self.model.reset()  # vllm的重置方法
            # else:
            #     self.model.clear_kv_cache()  # transformers的清理缓存方法
            
            # 也可以调用空闲缓存清理
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.debug("空闲缓存清理")

        if len(history) == 0:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = history + [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 添加token映射分析
        self.save_token_mapping(text)
        
        if not self.use_vllm:
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=actual_max_new_tokens,
                temperature=actual_temperature,
                top_p=actual_top_p
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in
                zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids,
                                                   skip_special_tokens=True)[0]

        else:
            # 更新采样参数，使用与初始化相同的参数结构
            self.sampling_params = SamplingParams(
                temperature=actual_temperature,
                top_p=actual_top_p,
                top_k=self.top_k if self.top_k > 0 else -1,
                repetition_penalty=1.0,
                max_tokens=actual_max_new_tokens,
                seed=self.seed,
                stop=None,
                ignore_eos=False
            )
            
            lora_request = None
            if self.lora_path:
                lora_request = LoRARequest(lora_name="mcloud", lora_int_id=1, lora_local_path=self.lora_path)
            
            outputs = self.model.generate([text], self.sampling_params, lora_request=lora_request)

            # 添加调试信息：打印输入token数量
            for output in outputs:
                prompt_token_ids = output.prompt_token_ids
                logger.debug(f"Input tokens count: {len(prompt_token_ids)}")
                logger.debug(f"Generated tokens count: {len(output.outputs[0].token_ids)}")
                # 如果需要更详细的信息，可以添加：
                # logger.debug(f"Prompt text: {output.prompt}")
                # logger.debug(f"Input token IDs: {prompt_token_ids}")
            
            response = ""
            for output in outputs:
                prompt = output.prompt
                response += output.outputs[0].text

        return response

    def _stream(
            self,
            prompt: str,
            history: List[Dict] = [],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        if len(history) == 0:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = history + [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        stop_tokens = self.tokenizer(stop, add_special_tokens=False)["input_ids"] if stop else []
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_tokens)])

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=self.max_new_token,
            temperature=self.temperature,
            top_p=self.top_p,
            stopping_criteria=stopping_criteria
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in
            zip(model_inputs.input_ids, generated_ids)
        ]

        for output_ids in generated_ids:
            for token_id in output_ids:
                token_text = self.tokenizer.decode(token_id, skip_special_tokens=True)
                # print(f"token_text = 【{token_text}】")
                if any(token_text.endswith(stop_seq) for stop_seq in stop or []):
                    break
                yield GenerationChunk(text=token_text)

    async def _astream(
            self,
            prompt: str,
            history: List[Dict] = [],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        if len(history) == 0:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = history + [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 所有可能耗时的操作（如tokenizer和model的调用）都使用await asyncio.to_thread()来异步执行。这样可以防止这些操作阻塞事件循环
        model_inputs = await asyncio.to_thread(
            lambda: self.tokenizer([text], return_tensors="pt").to(self.device)
        )

        stop_tokens = await asyncio.to_thread(
            lambda: self.tokenizer(stop, add_special_tokens=False)["input_ids"] if stop else []
        )
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_tokens)])

        # 异步生成token
        generation_kwargs = dict(
            model_inputs,
            streamer=self.streamer,
            max_new_tokens=self.max_new_token,
            temperature=self.temperature,
            top_p=self.top_p
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in self.streamer:
            yield GenerationChunk(text=new_text)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_new_token": self.max_new_token,
            "temperature": self.temperature,
            "top_p": self.top_p
        }


class Qwen2Remote(LLM):
    """Qwen LLM API service.

    Example:
        .. code-block:: python

            from qwen_langchain_service import Qwen
            qwen = Qwen(endpoint_url="http://192.168.32.113:7820/aibox/v1/llm/chat/completions")
    """

    endpoint_url: str = "http://192.168.32.113:7820/aibox/v1/llm/chat/completions"
    """Base URL of the Qwen API service."""
    max_tokens: int = 8000
    """Max token allowed to pass to the model."""
    temperature: float = 0.1
    """LLM model temperature from 0 to 10."""
    messages: List[dict] = []
    """History of the conversation"""
    top_p: float = 0.7
    """Top P for nucleus sampling from 0 to 1"""
    stream_flag: bool = False
    """Whether to use streaming chat or not"""
    model: str = "Qwen1.5-32B-Chat-GPTQ-Int4"
    """Qwen model to use"""
    with_history: bool = False
    """Whether to include the history in the response or not"""

    @property
    def _llm_type(self) -> str:
        return "qwen"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """Call out to a Qwen LLM inference endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = Qwen_llm("Who are you?")
        """

        # HTTP headers for authorization
        headers = {"Content-Type": "application/json"}

        if self.with_history:
            self.messages.append({"role": "user", "content": prompt})
        else:
            self.messages = [{"role": "user", "content": prompt}]

        payload = json.dumps({
            "request_id": "sassasaa11w1",
            "model": self.model,
            "messages": self.messages,
            "stream": self.stream_flag,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p
        })

        # logger.debug(f"Qwen payload: {payload}")

        # call api
        try:
            response = requests.request("POST", self.endpoint_url,
                                        headers=headers, data=payload)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        # logger.debug(f"Qwen response: {response}")

        if response.status_code != 200:
            raise ValueError(f"Failed with response: {response}")

        try:
            parsed_response = response.json()

            # Check if response content does exists
            if isinstance(parsed_response, dict):
                content_keys = "data"
                if content_keys in parsed_response:
                    text = parsed_response['data']['choices'][0]['message'][
                        'content']
                else:
                    raise ValueError(
                        f"No content in response : {parsed_response}")
            else:
                raise ValueError(f"Unexpected response type: {parsed_response}")

        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised during decoding response from inference endpoint: {e}."
                f"\nResponse: {response.text}"
            )

        return text

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        pass
