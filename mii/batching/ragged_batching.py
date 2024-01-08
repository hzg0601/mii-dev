# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import copy
import gc
import os
import queue
import random
import threading
import time
from collections import deque, defaultdict
from functools import cached_property
from typing import Dict, Tuple, List, Any, Union, DefaultDict

import torch
import ujson
import zmq
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.timer import SynchronizedWallClockTimer

from mii.batching.constants import (MAX_LENGTH_KWARG,
                                    MAX_NEW_TOKENS_KWARG,
                                    MIN_NEW_TOKENS_KWARG,
                                    STREAM_KWARG,
                                    IGNORE_EOS_KWARG,
                                    TOP_P_KWARG,
                                    TOP_K_KWARG,
                                    TEMPERATURE_KWARG,
                                    RETURN_FULL_TEXT_KWARG,
                                    DO_SAMPLE_KWARG,
                                    STOP_KWARG,
                                    MIN_NEW_TOKENS_DEFAULT,
                                    STREAM_DEFAULT,
                                    IGNORE_EOS_DEFAULT,
                                    TOP_P_DEFAULT,
                                    RETURN_FULL_TEXT_DEFAULT,
                                    DO_SAMPLE_DEFAULT,
                                    TOP_K_NAME,
                                    TOP_P_NAME,
                                    TEMP_NAME,
                                    SAMPLER_NAME,
                                    STOP_NAME)
from mii.batching.data_classes import Response, Request, RequestBatch
from mii.batching.generation.logit_processors import TopPLogitProcessor, TopKLogitProcessor, TemperatureLogitProcessor
from mii.batching.generation.samplers import LogitsSampler, GreedySampler
from mii.batching.generation.stop_criterion import EosGenerationStopCriterion, TokenStopCriterion
from mii.batching.postprocess import (
    run_batch_logit_processing,
    run_batch_sampler,
    run_batch_stop_criterion,
)
from mii.batching.utils import sync_debug, profiler
from mii.constants import GenerationFinishReason, ZMQ_RECV_TIMEOUT
from mii.logging import logger


class RaggedBatchBase:
    def __init__(self, inference_engine, tokenizer, model_config):
        self.inference_engine = inference_engine
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.model_config = model_config
        self.zmq_port = model_config.zmq_port_number
        if model_config.max_length is not None:
            self.max_length = model_config.max_length
        else:
            self.max_length = inference_engine._policy._checkpoint_engine.model_config.max_seq_length
        self.sync_debug = model_config.sync_debug
        self.profile_model_time = model_config.profile_model_time
        #? 定义请求队列，结果队列字典
        self.request_queue: queue.Queue = queue.Queue()
        #? 结果队列
        self.result_queues: Dict[int, queue.Queue] = {}
        #? 调度优化后的队列？
        # RequestBatch是Request的组合器，Request定义了to_msg_dict方法{"uid": self.uid, "input_tokens": input_tokens}
        # RequestBatch的to_msg_dict返回上述字典的list.
        self.scheduled_requests: RequestBatch = RequestBatch()
        self.buffer = deque()
        self.scheduled_length = 0
        self.scheduled_seq_num = 0
        #? 定义调度请求块的个数==kv cache group的个数
        self.scheduled_req_blocks = torch.zeros(inference_engine.n_kv_cache_groups,
                                                dtype=torch.int32,
                                                device="cpu")

        # TODO: we will need to prune self._post_processors for long running deployments
        self._post_processors = {}
        self.logit_processor = run_batch_logit_processing
        self.sampler = run_batch_sampler
        self.stop_criterion = run_batch_stop_criterion
        #? 同步墙时钟计时器
        self._timers: SynchronizedWallClockTimer = SynchronizedWallClockTimer()
        #? 耗时字典?
        self._profiled_times: DefaultDict[str, List[int]] = defaultdict(list)
        self._iters: int = 0
        self._num_generated_tokens: int = 0
        #* ZeroMQ是一种基于消息队列的多线程网络库,其对套接字类型、连接处理、帧、甚至路由的底层细节进行抽象,提供跨越多种传输协议的套接字。
        # 定义ZeroMQ的上下文管理器
        self._zmq_context = zmq.Context()
        # 初始化时进行cuda的同步
        torch.cuda.synchronize()
        # 对于host worker,创建zmq.PUB类socket,绑定tcp端口
        if self.is_rank_0:
            self.socket = self._zmq_context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{self.zmq_port}")
            time.sleep(1)  # Give the subscriber a change to connect
        # 对于非调度worker,创建SUB类socket,链接host worker的tcp地址
        # Set socket options
        else:
            self.socket = self._zmq_context.socket(zmq.SUB)
            self.socket.connect(f"tcp://localhost:{self.zmq_port}")
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.socket.setsockopt(zmq.RCVTIMEO, ZMQ_RECV_TIMEOUT)

    @cached_property
    def local_rank(self) -> int:
        return get_accelerator().current_device()

    @property
    def is_rank_0(self) -> bool:
        return self.local_rank == 0
    # 1. 调用_bcast_requests方法获取请求队列，按默认方式得到最小请求字典
    #       首先对于rank_0，如果self.scheduled_requests不为None&非force模式，直接返回self.schedule_requests,
    #       如果是force模式，则将self.schedule_requests转换为最小请求字典，转发给给其他rank;
    #       对于其他rank则直接尝试接受转发的最小request字典，如果不成功则调用RequestBatch类初始化该rank的
    #       self.schedule_requests
    # 2. 根据请求的uid列表，逐个调用self.inference_engine.flush方法flush `uid`,
    #?       该方法会flush请求队列，将已经完成的请求标记
    # 3. 如果scheduled_requests的requests_to_run（非flush请求列表构造的RequestBatch）不为空，则调用put方法
    #       将uid,和tokens作为输入，put方法会调用self.inference_engine.put方法
    #?       执行推理得到next_token_logits.执行完毕后，如果GPU不是rank_0,则直接返回，否则继续执行；
    # 4. 针对非flush的请求，将其从scheduled_requests队列中取出，调用update_seq_len方法更新，更新每个请求的seq_len+=len(input_tokens)
    #    如果更新后非flush的请求不为空，则调用self._process_logits方法，该方法调用
    #    self.logit_processor, self.sampler,self.stop_criterion得到next_tokens,以及
    #    next_tokens是否为stop_token的标识done_tokens，将得到的next_token,done_tokens
    #    作为最新值更新到running_requests中。
    #? 5. 调用self._reset_scheduler_bookkeeping初始化self.schedule_requests, 重新schedule requests队列???
    #    self.scheduled_length,self.scheduled_seq_num, self.scheduled_req_blocks为空或0
    # 6. 对于所有last_in_prompt为True的running_requests队列,调用请求的accumulate_generated_token，将
    #    next_token加入到_generated_tokens列表中，更新_num_generated_tokens，如果遇到请求的stop_generation标识
    #    或者strem标识，则将输出添加到self.result_queues字典，该字典key为请求的tid，该字典的值是一个Queue
    #    添加的方式是调用put_nowait方法；如果没有遇到请求的stop_generation标识，将生成的next_token作为
    #    input_token，并设置last_in_prompt为True, is_done为False
    #? 7. 取出已完成请求的id作为self.schedule_requests.prune的参数，将已完成的请求剔除，更新self.scheduled_requests
    #    调用self.schedule_requests方法，该方法的实现逻辑为：
            # 取出self.request_queue中所有不需等待的request,添加到buffer中，对于buffer中的每个request,
            # 如果该request为flush_request,则将该request添加到scheduled_requests中,
            # 如果不是flush_request且request的input_tokens长度为1，则将该request添加到next_token_gen_reqs列表中，
            # 如果不是flush_request且request的input_tokens长度>1，则将该request添加到prompt_reqs列表中.
            # 先调用_do_schedule_requests处理生成request,然后调用_do_schedule_requests处理prompt_req处理prefill request:
            # _do_schedule_requests的处理逻辑是：
                            # 在如下条件下，直接中断训练，不处理包含本请求的剩余请求序列：
                            #    如果inference_engine的free_blocks的最小值为0，
                            #    或者已经调度的token数，超过了conf_manager设定的token数，
                            #    或者调度后的请求数大于状态管理器设定的最大句子数，
                            #    或者请求是一个prefill请求，且输入的token数超过了查询引擎在该时刻允许处理的token数，
                            
                            # 在如下条件下，跳过某个请求的处理：
                            #    请求的的句子超度超出了该句子的总长度要求；
                            #    该请求是prefill请求，但推理引擎允许处理的req_tokens数大于请求的input_tokens数，或者是一个生成请求，
                            #      则重新查询该请求允许处理的token数，如果<=0,则跳过处理
                            
                            # 如果请求循环没有中断，或者请求没有被跳过，
                            #    则将该请求取出前req_tokens个加入scheduled_requests队列，更新self.scheduled_req_blocks，self.scheduled_length,
                            #    将该请求的剩余部分tokens从左侧加入buffer中。
            # 上述逻辑中会更新buffer和scheduled_requests队列，如果buffer不为空且scheduled_requests为空，则重新初始化scheduled_request,
            # 并调用reset_request_status,其处理逻辑为：
                #    取出旧buffer中reqeust的uid构造Request，以非阻塞的方式添加到request_queue中，
                #    将生成请求的token拼接到prompt token中作为新的请求, 将这些请求都添加到buffer中作为新的buffer；
            # 如果 buffer不为空&scheduled_requests为空 不成立，即若buffer为空或(且)schedule_requests不为空，
            #   取出所有self.scheduled_requests的id作为一个集合，将不在scheduled_requests中的请求作为新的buffer.            
    # 如果self._profile_model_times，则调用_print_profiled_times, 
    #   每次都调用该函数来计数，每100个轮次打印一次，非生成任务，直接打印，生成任务额外打印生成 tokens/ms,打印完毕后清零。
    @profiler
    def generate(self) -> None:
        # 1. Get a batch of requests, broadcast to all ranks
        scheduled_requests = self._bcast_requests()

        # 2. Flush for uids that are finished generating
        self.flush(scheduled_requests.requests_to_flush.uids)

        # 3. Put new tokens into inference engine
        if scheduled_requests.requests_to_run:
            next_token_logits = self.put(
                scheduled_requests.requests_to_run.uids,
                scheduled_requests.requests_to_run.tokens,
            )

        # short circuit if not rank 0, only rank 0 does scheduling and postprocessing of logits
        if not self.is_rank_0:
            return

        # 4. Launch logit processing and token generation
        running_requests = scheduled_requests.requests_to_run
        running_requests.update_seq_length()
        if running_requests:
            next_tokens, done_tokens = self._process_logits(
                next_token_logits, running_requests
            )
            running_requests.next_tokens = next_tokens
            running_requests.done_tokens = done_tokens

        #? 5. Schedule requests while we wait for the forward pass to finish
        self._reset_scheduler_bookkeeping()

        # 6. Accumulate generated tokens, check completion, and generate output
        for r in running_requests.last_in_prompt:
            r.accumulate_generated_token()
            self._num_generated_tokens += 1
            if r.stop_generation or r.stream:
                self._generate_output(r)
            if not r.stop_generation:
                r.set_next_as_input()
                self.request_queue.put(r)

        # 7. Update scheduled requests
        self.scheduled_requests.prune(running_requests.completed.uids)
        self.schedule_requests()

        if self.profile_model_time:
            self._print_profiled_times()
    # 每次都调用该函数来计数，每100个轮次打印一次，非生成任务，直接打印，生成任务额外打印生成 tokens/ms 
    # 100个轮次打印完毕后清零
    def _print_profiled_times(self) -> None:
        self._iters += 1
        if not (self._iters % 100 == 0):
            return
        for event, times in self._profiled_times.items():
            mean_time = sum(times) / len(times)
            log_msg = f"{event}: {mean_time}"
            if event == "generate":
                log_msg += f" ({self._num_generated_tokens / sum(times)} tokens/ms)"
            logger.info(log_msg)
        self._profiled_times.clear()
        self._num_generated_tokens = 0
    # Rank 0 gets batch of requests and broadcasts to other ranks
    @sync_debug
    def _bcast_requests(self, force=False) -> RequestBatch:
        # 对于host worker, 如果调度后请求为空，且非force模式，则返回调度后请求；
        # scheduled_requests的元素为{"uid": self.uid, "input_tokens": input_tokens}
        # 否则，调用to_msg_dict将request的最小版本构成字典的list，转换为json_str,调用socket广播到各个worker上
        if self.is_rank_0:
            if not self.scheduled_requests and not force:
                return self.scheduled_requests
            # Rank 0 gets batch of requests and broadcasts to other ranks
            # to_msg_dict Returns a minimal version of the request of purposes of broadcasting to all ranks
            data_dicts = self.scheduled_requests.to_msg_dicts()
            json_data = ujson.dumps(data_dicts)
            self.socket.send_string(json_data)
        # 对于server worker, 首先调用socker.recv_string方法接受最小request version，
        # 如果出错，则将scheduled_requests重新初始化
        else:
            try:
                json_data = self.socket.recv_string()
                data_dicts = ujson.loads(json_data)
                self.scheduled_requests = RequestBatch.from_msg_dicts(data_dicts)
            except zmq.Again:
                self.scheduled_requests = RequestBatch()

        return self.scheduled_requests

    def _reset_scheduler_bookkeeping(self) -> None:
        self.scheduled_requests = RequestBatch()
        self.scheduled_length = 0
        self.scheduled_seq_num = 0
        self.scheduled_req_blocks.zero_()

    @sync_debug
    def _process_logits(
            self,
            next_token_logits: torch.Tensor,
            running_requests: RequestBatch) -> Tuple[torch.Tensor,
                                                     torch.Tensor]:
        next_token_logits = next_token_logits[:, :self.vocab_size]
        next_token_logits = self.logit_processor(next_token_logits,
                                                 running_requests,
                                                 self._post_processors)
        next_tokens = self.sampler(next_token_logits,
                                   running_requests,
                                   self._post_processors)
        done_tokens = self.stop_criterion(next_tokens,
                                          running_requests,
                                          self._post_processors)
        next_tokens = next_tokens.to(torch.device("cpu"), non_blocking=False)
        return next_tokens, done_tokens
    # uid 请求的唯一ID，tid是请求所在的thread唯一ID，以tid作为输出的key
    @sync_debug
    def _generate_output(self, r: Request) -> bool:
        outputs = []
        if r.stream:
            outputs.append((
                r.uid,
                [r.next_token],
                r.prompt_length,
                r.num_generated_tokens,
                GenerationFinishReason.NONE,
            ))
        if r.finish_reason != GenerationFinishReason.NONE:
            if r.stream or not r.generated_tokens:
                output_tokens = []
            else:
                output_tokens = torch.cat([t.unsqueeze(0) for t in r.generated_tokens],
                                          dim=0)
                if r.return_full_text:
                    # Avoid returning bos token, refactor this later
                    output_tokens = torch.cat((r.prompt_tokens[1:], output_tokens))
            outputs.append((
                r.uid,
                output_tokens,
                r.prompt_length,
                r.num_generated_tokens,
                r.finish_reason,
            ))
        for output in outputs:
            self.result_queues[r.tid].put_nowait(output)
    # 在如下条件下，直接中断循环，不处理包含本请求的剩余请求序列：
    #    如果inference_engine的free_blocks的最小值为0，
    #    或者已经调度的token数，超过了conf_manager设定的token数，
    #    或者调度后的请求数大于状态管理器设定的最大句子数，
    #    或者请求是一个prefill请求，且输入的token数超过了查询引擎在该时刻允许处理的req_token数，
    
    # 在如下条件下，跳过某个请求的处理：
    #    请求的的句子超度超出了该句子的总长度要求；
    #    该请求是prefill请求，但推理引擎允许处理的req_tokens数大于请求的input_tokens数，或者是一个生成请求，
    #      则重新查询该请求允许处理的token数，如果<=0,则跳过处理；
    
    # 如果请求循环没有中断，或者请求没有被跳过，
    #    则将该请求取出前req_tokens个加入调度请求队列，更新self.scheduled_req_blocks，self.scheduled_length,
    #    将该请求的剩余部分tokens从左侧加入buffer中。
    def _do_schedule_requests(self, requests: List[Request]) -> None:
        #* inference_engine上每个序列所剩余的kv cache blocks数，每个kv cache block包含若干kv cache对
        # the number of free blocks in each cache
        # inference_engine.free_blocks的调用链，返回_state_manager: DSStateManager.free_blocks -> 
        #   DSStateManager._kv_cache: BlockedKVCache.free_blocks->
        #   BlockedKVCache.free_blocks -> torch.empty(len(self._allocators))元素为BlockedAllocator(num_blocks).free_blocks ->
        #   BlockedAllocator(num_blocks).free_blocks的类型为int,指示该kv cache内的空余blocks数
        # 故综上，free_blocks为一个元素为各个kv-cache num_free_blocks的Tensor
        free_blocks = self.inference_engine.free_blocks
        conf_manager = self.inference_engine._config.state_manager
        print("*"*108)
        print(f"the self.inference_engine is:{self.inference_engine}")
        print(f"the self.inference_engine._config is:{self.inference_engine._config}")
        print(f"the self.inference_engine.free_blocks is:{self.inference_engine.free_blocks}")
        print(f"the self.inference_engine._config.state_manager is:{self.inference_engine._config.state_manager}")
        print(f"the self.scheduled_requests is:{self.scheduled_requests}")
        print(f"the self.scheduled_length is:{self.scheduled_length}")
        print(f"the self.scheduled_req_blocks is:{self.scheduled_req_blocks}")
        for r in requests:
            #? 如果free_blocks最小值为0，则中断循环，不再处理剩余请求；->某个句子的prefill阶段处理完毕?
            # 如果请求的序列长度超过设定的最大长度，则跳过该请求不再处理
            # 如果调度后的请求数大于状态管理器设定的max_ragged_sequence_count,则中断循环，不再处理剩余请求
            #* 如果剩余允许max_batch_size<=0,则中断循环，不再处理剩余请求->已经调度的token数，超过了conf_manager设定的token数

            #? max_ragged_sequence_count一个batch被分割后的所有chunk数？inference_engine._config的最大处理句子数？
            #? scheduled_length-> schedule_requests的token数?
            #? max_ragged_batch_size->batch内允许的最大token数?
            if free_blocks.min().item() == 0:
                break

            if r.max_length <= r.seq_length:
                continue

            # Make sure that the engine has enough capacity to process the batch
            if len(self.scheduled_requests) > conf_manager.max_ragged_sequence_count:
                break
            # self.scheduled_length是一个指示token的数，故conf_manager.max_ragged_batch_size也是一个数，max_batch_size也是一个数
            max_batch_size = conf_manager.max_ragged_batch_size - self.scheduled_length
            if max_batch_size <= 0:
                break
            #? ------------------------------------------------
            # 定义剩余的max_blocks = free_blocks - scheduled_req_blocks,如果请求的input_token>1即请求是prefill请求
            # 则调用engine.query方法查询给定input_token长度和剩余max_blocks限制下的req_tokens数，
            # 如果req_tokens数小于请求的input_tokens数，即请求的prompt长度超过允许的req_tokens长度，则中断循环，不再处理剩余请求；
            # 如果req_tokens数大于请求的input_tokens数，或者该请求的input_tokens==1，即是一个生成请求，
            #    重新设定req_tokens为请求输入token数和batch剩余最大token数max_batch_size的最小值
            # 调用engine.query方法查询给定req_tokens长度和剩余max_blocks限制下的req_tokens数和req_blocks
            # 如果允许的req_tokens小于0，则跳过该请求
            #! free_blocks是一个Tensor, self.scheduled_req_blocks也是一个Tensor，故max_blocks是一个Tensor
            # self.scheduled_req_blocks = torch.zeros(inference_engine.n_kv_cache_groups）
            max_blocks = free_blocks - self.scheduled_req_blocks

            # Check capacity to mitigate the deadlock risk
            # We don't schedule requests when we find that a prompt is too long to fit to the KV cache
            if len(r.input_tokens) > 1:
                req_tokens, _ = self.inference_engine.query(r.uid, len(r.input_tokens), max_blocks)
                print(f"the req_tokens after self.inference_engine.query(r.uid, len(r.input_tokens), max_blocks) is: {req_tokens}")
                if req_tokens < len(r.input_tokens):
                    break
            #!!! req_tokens并不是固定的，是input_tokens和conf_manager允许处理的最小值
            req_tokens = min(len(r.input_tokens), max_batch_size)
            # Given a sequence and the number of new tokens in the sequence, determine the
            # number of new KV blocks needed to support the sequence. This method is
            # used to help the engine provide schedulability APIs and can be used as a helper
            # for ``maybe_allocate_kv``.
            # inference_engine.query会把uid从_state_manager中找到完整的token序列，然后调用inference_engine.get_kv_requirements方法
            #* req_tokens数+已有token数seen_tokens//inference_engine.attn.kv_block_size(每个block的kv对数量)得到本序列需要请求的req_blocks
            #* req_blocks-序列当前已分配的blocks数cur_allocated_blocks即需要额外新增的block数block_lim,如果block_lim小于max_blocks,则返回max_new_tokens和block_lim
            #* 否则计算(max_block+cur_allocated_blocks)*inference_engine.attn.kv_block_size-seen_tokens得到当前最大允许的token数token_capacity
            #* 返回token_capacity, torch.tensor([max_new_blocks]) 
            #*! 即如果空余的max_blocks充足，则req_tokens不变，req_blocks变少（按需分配）；如果空余的max_blocks不足，则req_blocks不变，req_tokens变少（按剩多少block分配）
            # cur_allocated_blocks: int, the number of blocks currently allocated for this sequence in the specified cache group
            #???? bug report In inference_model_base.py, the `DSInferenceModelBase.get_kv_requirements` requires the `max_new_blocks` be a Tuple or something like Tuple at least.
            #???? However, in inference_transformer_base.py, the `DSTransformerModelBase`, as the children class of `DSInferenceModelBase`, its realization of `get_kv_requirements`
            #???? requires the `max_new_blocks` be a Int. I carefully checked the parameter `free_blocks` of `DSTransformerModelBase`, eventually, it's a Tensor, which could be
            #???? interpreated as a Tuple according with `DSInferenceModelBase`. Since all concrete models (such as opt, llama) are children classes of `DSTransformerModelBase`,
            #???? when we call `RaggedBatchBase._do_schedule_requests` thereby call `self.inference_engine.query`, we actually call  `DSTransformerModelBase.get_kv_requirements`,which
            #???? requires the `max_new_blocks` be a Int. But, again, in `RaggedBatchBase._do_schedule_requests`, `free_blocks` is a Tensor, `max_blocks = free_blocks - self.scheduled_req_blocks`
            #???? must be a Tensor too, which does not meet `self.inference_engine.query`. Is it a mistake, or there is some key technique I missed?  Do please help me figure out it.
            req_tokens, req_blocks = self.inference_engine.query(r.uid, req_tokens, max_blocks)
            print(f"the req_tokens,req_blocks after self.inference_engine.query(r.uid, req_tokens, max_blocks) is: {req_tokens},{req_blocks}")
            if req_tokens <= 0:
                continue

            # Decompose the prompt to fit to the max ragged batch size
            # 更新本次输入的tokens数为req_tokens数，如果本次请求的tokens数大于输入的tokens数
            # 则指定请求的last_in_prompt为True
            # 将本次请求加入scheduled_requests队列，基于本次处理的req_blocks和req_tokens
            # 更新scheduled_req_blocks、scheduled_length
            # 如果本次请求不是最后一次prefill请求，则将该请求作为req_remaining,
            # 以剩余token为input_token,以seq_length+req_tokens为新的seq_length,令请求的last_in_prompt为True
            # 将更新的req_remaining添加到buffer的左端。
            #? scheduled_req_blocks是一个列表，scheduled_length是一个数？
            # req_tokens小于input_tokens,
            decomposed = req_tokens < len(r.input_tokens)
            remaining_tokens = r.input_tokens[req_tokens:]
            r.input_tokens = r.input_tokens[:req_tokens]
            r.last_in_prompt = not decomposed # gen请求的last_in_prompt必然

            # Schedule the request
            self.scheduled_requests.append(r)
            #!?! scheduled_req_blocks, req_blocks, max_blocks, free_blocks是否是同一类型？
            self.scheduled_req_blocks += req_blocks
            self.scheduled_length += req_tokens # req_tokens是一个数，故scheduled_length是一个数

            if decomposed:
                req_remaining = copy.copy(r)
                req_remaining.input_tokens = remaining_tokens
                req_remaining.seq_length = r.seq_length + req_tokens # req_tokens是一个数,故seq_length是一个数，
                req_remaining.last_in_prompt = True

                self.buffer.appendleft(req_remaining)
    # 取出所有不需等待的request,添加到buffer中，对于buffer中的每个request：
        # 如果该request为flush_request,则将该request添加到scheduled_requests中,
        # 如果不是flush_request且request的input_tokens长度为1，则将该request添加到next_token_gen_reqs列表中，
        # 如果不是flush_request且request的input_tokens长度>1，则将该request添加到prompt_reqs列表中.
        #*（此时self.buffer保留所有no_wait请求，scheduled_requests中添加了flush_request,request_queue移除了所有no_wait请求）
        # 先调用_do_schedule_requests处理生成request,然后调用_do_schedule_requests处理prompt_req处理prefill request
    # _do_schedule_requests的处理逻辑是：
                    # 在如下条件下，直接中断训练，不处理包含本请求的剩余请求序列：
                    #    如果inference_engine的free_blocks的最小值为0，
                    #    或者已经调度的token数，超过了conf_manager设定的token数，
                    #    或者调度后的请求数大于状态管理器设定的最大句子数，
                    #    或者请求是一个prefill请求，且输入的token数超过了查询引擎在该时刻允许处理的req_tokens数，
                    
                    # 在如下条件下，跳过某个请求的处理：
                    #    请求的的句子超度超出了该句子的总长度要求；
                    #    该请求是prefill请求，但推理引擎允许处理的req_tokens数大于请求的input_tokens数，或者是一个生成请求，
                    #      则重新查询该请求允许处理的token数，如果<=0,则跳过处理
                    
                    # 如果请求循环没有中断，或者请求没有被跳过，(prefill请求且请求的req_token数大于input_tokens,或者gen请求且允许请求req_tokens数大于0)
                    #    则将该请求取出前req_tokens个加入scheduled_requests队列，更新self.scheduled_req_blocks，self.scheduled_length,
                    #    将该请求的剩余部分tokens从左侧加入buffer中。
    # 上述逻辑中是更新buffer和scheduled_requests队列，如果buffer不为空且scheduled_requests为空，则重新初始化scheduled_request,
    # 并调用reset_request_status,其处理逻辑为：
        #    取出旧buffer中reqeust的uid构造Request，以非阻塞的方式添加到request_queue中，
        #    将生成请求的token拼接到prompt token中作为新的请求, 将这些请求都添加到buffer中作为新的buffer；
    # 如果 buffer不为空&scheduled_requests为空 不成立，即若buffer为空或(且)schedule_requests不为空，
    #   取出所有scheduled_requests的id作为一个集合，将不在scheduled_requests中的请求作为新的buffer.
    def schedule_requests(self) -> None:
        # 如果request_queue不为空，则取出所有不需等待的request,添加到buffer中
        while not self.request_queue.empty():
            r = self.request_queue.get_nowait()
            self.buffer.append(r)

        # Run next token generation first
        next_token_gen_reqs = []
        prompt_reqs = []
        # 对于buffer中的每个request,如果该request为flush_request,
        # 则将该request添加到scheduled_requests中,否则
        # 如果request的input_tokens长度为1，则将该request添加到next_token_gen_reqs列表中，
        # 如果request的input_tokens长度>1，则将该request添加到prompt_reqs列表中.
        for r in self.buffer:
            if r.is_flush_request:
                self.scheduled_requests.append(r)
            else:
                if len(r.input_tokens) == 1:
                    next_token_gen_reqs.append(r)
                else:
                    prompt_reqs.append(r)

        # We want to process next token generation first
        #* 先调用_do_schedule_requests处理生成request,然后调用_do_schedule_requests处理prompt_req处理prefill-----
        # 如果buffer不为空，但scheduled_requests为空，则重新初始化scheduled_request,
        # 并调用reset_request_status,取出旧buffer中reqeust的uid构造Request，以非阻塞的方式添加到request_queue中，
        # 将生成请求的token拼接到prompt token中作为新的请求, 将这些请求都添加到buffer中作为新的buffer；
        # 否则，取出所有scheduled_requests的id作为一个集合，将不在scheduled_requests中的请求作为新的buffer.
        print("----------------------------------start the next_token_gen_reqs loop.........")
        self._do_schedule_requests(next_token_gen_reqs)
        print("-----------------------------------start the prompt_reqs loop.........")
        self._do_schedule_requests(prompt_reqs)

        if len(self.buffer) > 0 and len(self.scheduled_requests) == 0:
            print(
                "Deadlock detected. Resetting KV cache and recomputing requests. Consider limiting number of concurrent requests or decreasing max lengths of prompts/generations."
            )
            self.scheduled_requests = RequestBatch()
            self.reset_request_status()
        else:
            scheduled_requests_ids = set(id(r) for r in self.scheduled_requests)
            self.buffer = deque(
                [r for r in self.buffer if id(r) not in scheduled_requests_ids])
    # 将Request以非阻塞的方式添加到request_queue中，如果没有槽位会raise Full exception
    # 该Request只有uid是非None，其他均为None.
    def _queue_flush_request(self, uid: int) -> None:
        # Put an item into the queue without blocking.
        # Only enqueue the item if a free slot is immediately available.
        # Otherwise raise the Full exception.
        self.request_queue.put_nowait(
            Request(
                tid=None,
                uid=uid,
                input_tokens=None,
                prompt_tokens=None,
                seq_length=None,
                max_length=None,
                max_new_tokens=None,
                min_new_tokens=None,
                last_in_prompt=None,
                post_processing=None,
                stream=None,
            ))
    #* 取出旧buffer中reqeust的uid构造Request，以非阻塞的方式添加到request_queue中，
    #* 将生成的请求的token拼接到prompt token中作为新的prefill请求, 将这些请求都添加到buffer中
    
    def reset_request_status(self):
        # 针对buffer里的每个request,如果request的seq_length不为0，则
        # 取出reqeust的uid构造Request，以非阻塞的方式添加到request_queue中
        for r in self.buffer:
            if r.seq_length > 0:
                self._queue_flush_request(r.uid)
        # deque, list-like container with fast appends and pops on either end
        # 先定义一个新的buffer,
        # 对旧buffer中的每个request,先复制一份作为新的request，然后将request的prompt_token和generated_tokens合并
        # 作为新的prompt_token和input_token,更新seq_length, max_new_tokens,删除generated_tokens
        # 然后把新的request添加到新的buffer中，作为类的buffer.
        new_buffer = deque()
        for r in self.buffer:
            new_req = copy.copy(r)
            new_req.prompt_tokens = new_req.input_tokens = torch.concat(
                [r.prompt_tokens] + [t.unsqueeze(0) for t in r.generated_tokens])
            new_req.seq_length = 0
            new_req.max_new_tokens = r.max_new_tokens - len(r.generated_tokens)
            new_req.clear_generated_token()
            new_buffer.append(new_req)

        self.buffer = new_buffer
    # 根据input_tokens和generation token构造Request
    def make_request(self,
                     tid: int,
                     uid: int,
                     input_tokens: torch.Tensor,
                     kwargs: Dict) -> Request:
        prompt_length = len(input_tokens)
        # max_length是输入+输出的最大允许长度
        max_length = kwargs.pop(MAX_LENGTH_KWARG, self.max_length)
        assert max_length > prompt_length, f"prompt length must be less than {MAX_LENGTH_KWARG}"
        max_new_tokens = kwargs.pop(MAX_NEW_TOKENS_KWARG, max_length - prompt_length)
        min_new_tokens = kwargs.pop(MIN_NEW_TOKENS_KWARG, MIN_NEW_TOKENS_DEFAULT)
        stream = kwargs.pop(STREAM_KWARG, STREAM_DEFAULT)
        ignore_eos = kwargs.pop(IGNORE_EOS_KWARG, IGNORE_EOS_DEFAULT)
        return_full_text = kwargs.pop(RETURN_FULL_TEXT_KWARG, RETURN_FULL_TEXT_DEFAULT)

        post_processing = []

        top_p = kwargs.pop(TOP_P_KWARG, TOP_P_DEFAULT)
        top_p_name = "_".join((TOP_P_NAME, str(top_p)))
        if top_p_name not in self._post_processors:
            self._post_processors[top_p_name] = TopPLogitProcessor(top_p=top_p)
        post_processing.append(top_p_name)

        top_k = kwargs.pop(TOP_K_KWARG, None)
        if top_k is not None:
            top_k_name = "_".join((TOP_K_NAME, str(top_k)))
            if top_k_name not in self._post_processors:
                self._post_processors[top_k_name] = TopKLogitProcessor(top_k=top_k)
            post_processing.append(top_k_name)

        temp = kwargs.pop(TEMPERATURE_KWARG, None)
        if temp is not None:
            temp_name = "_".join((TEMP_NAME, str(temp)))
            if temp_name not in self._post_processors:
                self._post_processors[temp_name] = TemperatureLogitProcessor(
                    temperature=temp)
            post_processing.append(temp_name)

        do_sample = kwargs.pop(DO_SAMPLE_KWARG, DO_SAMPLE_DEFAULT)
        if do_sample:
            sampler_name = "_".join((SAMPLER_NAME, "logits"))
            if sampler_name not in self._post_processors:
                self._post_processors[sampler_name] = LogitsSampler()
        else:
            sampler_name = "_".join((SAMPLER_NAME, "greedy"))
            if sampler_name not in self._post_processors:
                self._post_processors[sampler_name] = GreedySampler()
        post_processing.append(sampler_name)

        stop = kwargs.pop(STOP_KWARG, None)
        if stop is not None:
            stop_name = "_".join((STOP_NAME, stop))
            if stop_name not in self._post_processors:
                self._post_processors[stop_name] = TokenStopCriterion(
                    token=stop,
                    tokenizer=self.tokenizer)
        else:
            stop_name = STOP_NAME
            if STOP_NAME not in self._post_processors:
                self._post_processors[stop_name] = EosGenerationStopCriterion(
                    tokenizer=self.tokenizer)
        post_processing.append(stop_name)

        assert kwargs == {}, f"Unknown keyword arguments {kwargs}"

        return Request(
            tid=tid,
            uid=uid,
            input_tokens=input_tokens,
            prompt_tokens=input_tokens,
            seq_length=0,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            last_in_prompt=True,
            post_processing=post_processing,
            stream=stream,
            ignore_eos=ignore_eos,
            return_full_text=return_full_text,
        )
    # 以Response类的形式返回答案
    def make_response(self,
                      generated_text: str,
                      prompt_length: int,
                      generated_length: int,
                      finish_reason: GenerationFinishReason) -> Response:
        return Response(generated_text=generated_text,
                        prompt_length=prompt_length,
                        generated_length=generated_length,
                        finish_reason=finish_reason)
    # 调用inference_engine的put方法将uids,tokenized_input作为request
    def put(self, uids: List[int], tokenized_input: List[torch.Tensor]) -> torch.Tensor:
        return self.inference_engine.put(uids, tokenized_input)
    #? 调用inference_engine.flush函数刷新uid
    def flush(self, uids: List[int]) -> None:
        for uid in uids:
            self.inference_engine.flush(uid)


class MIIPipeline(RaggedBatchBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Return a non-zero integer that uniquely identifies the current thread amongst other threads that exist simultaneously
        self.tid = threading.get_ident()
        self._destroyed = False
    # 1. 调用self._put_request方法初始化self.request_queue和self.request_queue
        # 基于self.tid初始化result_queues字典，值为Queue
        # 调用self.tokenizer经input_str转换为token
        # 调用self.make_request方法构造一个Request示例request
        # 调用Queue的put方法将request放入self.request_queue
    # 2. 调用self.schedule_requests方法
            # 取出request_queue中所有不需等待的request,添加到buffer中，对于buffer中的每个request,
            # 如果该request为flush_request,则将该request添加到scheduled_requests中,
            # 如果不是flush_request且request的input_tokens长度为1，则将该request添加到next_token_gen_reqs列表中，
            # 如果不是flush_request且request的input_tokens长度>1，则将该request添加到prompt_reqs列表中.
            #*（此时self.buffer保留所有no_wait请求，scheduled_requests中添加了flush_request,request_queue移除了所有no_wait请求）
            # 先调用_do_schedule_requests处理gen request, ()
            # 然后调用_do_schedule_requests处理prefill request:
            # _do_schedule_requests的处理逻辑是：
                            # 在如下条件下，直接中断训练，不处理包含本请求的剩余请求序列：
                            #    如果inference_engine的free_blocks的最小值为0，
                            #    或者已经调度的token数，超过了conf_manager设定的token数，
                            #    或者调度后的请求数大于状态管理器设定的最大句子数，
                            #    或者请求是一个prefill请求，且输入的token数超过了查询引擎在该时刻允许处理的req_token数，
                            
                            # 在如下条件下，跳过某个请求的处理：
                            #    请求的的句子超度超出了该句子的总长度要求；
                            #    该请求是prefill请求，但推理引擎允许处理的req_tokens数大于请求的input_tokens数，或者是一个gen请求，
                            #      则重新查询该请求允许处理的token数，如果<=0,则跳过处理
                            
                            # 如果请求循环没有中断，或者请求没有被跳过，(prefill请求且请求的req_token数大于input_tokens,或者gen请求且允许请求req_tokens数大于0)
                            #    则将该请求取出前req_tokens个加入scheduled_requests队列，更新self.scheduled_req_blocks，self.scheduled_length,
                            #    将该请求的剩余部分tokens从左侧加入buffer中。
            # 上述逻辑中是更新buffer和scheduled_requests队列，如果buffer不为空且scheduled_requests为空，则重新初始化scheduled_request,
            # 并调用reset_request_status,其处理逻辑为：
                #    取出旧buffer中reqeust的uid构造Request，以非阻塞的方式添加到request_queue中，
                #    将生成请求的token拼接到prompt token中作为新的请求, 将这些请求都添加到buffer中作为新的buffer；
            # 如果 buffer不为空&scheduled_requests为空 不成立，即若buffer为空或(且)schedule_requests不为空，
            #   取出所有scheduled_requests的id作为一个集合，将不在scheduled_requests中的请求作为新的buffer.
    # 3. 在rank_0上，将所有uid按长度的range构造为一个列表，只要列表不为空，则执行：
            # 调用self.generate方法进行生成，一次生成后，
                        # 1. 调用_bcast_requests方法获取请求队列，按默认方式得到最小请求字典
                        #       首先对于rank_0，如果self.scheduled_requests不为None&非force模式，直接返回self.schedule_requests,
                        #       如果是force模式，则将self.schedule_requests转换为最小请求字典，转发给给其他rank;
                        #       对于其他rank则直接尝试接受转发的最小request字典，如果不成功则调用RequestBatch类初始化该rank的
                        #       self.schedule_requests
                        # 2. 根据请求的uid列表，逐个调用self.inference_engine.flush方法flush `uid`,
                        #?       该方法会flush请求队列，将已经完成的请求标记
                        # 3. 如果scheduled_requests的requests_to_run（非flush请求列表构造的RequestBatch）不为空，则调用put方法
                        #       将uid,和tokens作为输入，put方法会调用self.inference_engine.put方法
                        #?       执行推理得到next_token_logits.执行完毕后，如果GPU不是rank_0,则直接返回，否则继续执行；
                        # 4. 针对非flush的请求，将其从scheduled_requests队列中取出，调用update_seq_len方法更新，更新每个请求的seq_len+=len(input_tokens)
                        #    如果更新后非flush的请求不为空，则调用self._process_logits方法，该方法调用
                        #    self.logit_processor, self.sampler,self.stop_criterion得到next_tokens,以及
                        #    next_tokens是否为stop_token的标识done_tokens，将得到的next_token,done_tokens
                        #    作为最新值更新到running_requests中。
                        #? 5. 调用self._reset_scheduler_bookkeeping初始化self.schedule_requests, 重新schedule requests队列???
                        #    self.scheduled_length,self.scheduled_seq_num, self.scheduled_req_blocks为空或0
                        # 6. 对于所有last_in_prompt为True的running_requests队列,调用请求的accumulate_generated_token，将
                        #    next_token加入到_generated_tokens列表中，更新_num_generated_tokens，如果遇到请求的stop_generation标识
                        #    或者strem标识，则将输出添加到self.result_queues字典，该字典key为请求的tid，该字典的值是一个Queue
                        #    添加的方式是调用put_nowait方法；如果没有遇到请求的stop_generation标识，将生成的next_token作为
                        #    input_token，并设置last_in_prompt为True, is_done为False
                        #? 7. 取出已完成请求的id作为self.schedule_requests.prune的参数，将已完成的请求剔除，更新self.scheduled_requests
                        #    调用self.schedule_requests方法，该方法的实现逻辑为：
                                # 取出self.request_queue中所有不需等待的request,添加到buffer中，对于buffer中的每个request,
                                # 如果该request为flush_request,则将该request添加到scheduled_requests中,
                                # 如果不是flush_request且request的input_tokens长度为1，则将该request添加到next_token_gen_reqs列表中，
                                # 如果不是flush_request且request的input_tokens长度>1，则将该request添加到prompt_reqs列表中.
                                # 先调用_do_schedule_requests处理生成request,然后调用_do_schedule_requests处理prompt_req处理prefill request:
                                # _do_schedule_requests的处理逻辑是：
                                                # 在如下条件下，直接中断训练，不处理包含本请求的剩余请求序列：
                                                #    如果inference_engine的free_blocks的最小值为0，
                                                #    或者已经调度的token数，超过了conf_manager设定的token数，
                                                #    或者调度后的请求数大于状态管理器设定的最大句子数，
                                                #    或者请求是一个prefill请求，且输入的token数超过了查询引擎在该时刻允许处理的token数，
                                                
                                                # 在如下条件下，跳过某个请求的处理：
                                                #    请求的的句子超度超出了该句子的总长度要求；
                                                #    该请求是prefill请求，但推理引擎允许处理的req_tokens数大于请求的input_tokens数，或者是一个生成请求，
                                                #      则重新查询该请求允许处理的token数，如果<=0,则跳过处理
                                                
                                                # 如果请求循环没有中断，或者请求没有被跳过，
                                                #    则将该请求取出前req_tokens个加入scheduled_requests队列，更新self.scheduled_req_blocks，self.scheduled_length,
                                                #    将该请求的剩余部分tokens从左侧加入buffer中。
                                # 上述逻辑中会更新buffer和scheduled_requests队列，如果buffer不为空且scheduled_requests为空，则重新初始化scheduled_request,
                                # 并调用reset_request_status,其处理逻辑为：
                                    #    取出旧buffer中reqeust的uid构造Request，以非阻塞的方式添加到request_queue中，
                                    #    将生成请求的token拼接到prompt token中作为新的请求, 将这些请求都添加到buffer中作为新的buffer；
                                # 如果 buffer不为空&scheduled_requests为空 不成立，即若buffer为空或(且)schedule_requests不为空，
                                #   取出所有self.scheduled_requests的id作为一个集合，将不在scheduled_requests中的请求作为新的buffer.            
                        # 如果self._profile_model_times，则调用_print_profiled_times, 
                        #   每次都调用该函数来计数，每100个轮次打印一次，非生成任务，直接打印，生成任务额外打印生成 tokens/ms,打印完毕后清零。
            # 若result_queue队列中该线程Queue不为空，
                # 则调用_get_reponse方法，将结果取出，解码，调用make_response方法将结果组装为Response类
                # 将结果添加到outputs列表中
                # 调用lsef._queue_flush_request将uid组装为Request示例以非阻塞的方式添加到request_queue中
                # 将该请求在uids_running中的index添加到uids_complete_order列表中，然后将该请求从uids_running中移除
        # 4. 处理完所有uids_running后，调用self._bcast_requests，将scheduled_requests队列广播到其他rank上；
    # 5. 如果当前worker不是rank_0,则调用self.generate方法，直到处理完所有scheduled_requests
    # 6. 处理完毕后，将输出按照uids_running中的顺序进行排列
    #    如果要求所有rank上都输出，则调用_bcast_requests将结果进行广播，最后返回结果
        
    def __call__(self, inputs: Union[str, List[str]], **kwargs) -> List[Response]:
        if self._destroyed:
            raise RuntimeError(
                "The inference engine of this pipeline has been destroyed.")

        if isinstance(inputs, str):
            inputs = [inputs]
        outputs: List[Response] = []
        uids_running: List[int] = list(range(len(inputs)))
        uids_complete_order: List[int] = []

        for uid, input in zip(uids_running, inputs):
            request_kwargs = kwargs.copy()
            self._put_request(uid, input, request_kwargs)

        self.schedule_requests()

        if self.is_rank_0:
            # Rank 0 runs generate() until all responses are returned
            while uids_running:
                self.generate()
                while not self.result_queues[self.tid].empty():
                    uid, response = self._get_response()
                    outputs.append(response)
                    self._queue_flush_request(uid)
                    uids_complete_order.append(uids_running.index(uid))
                    uids_running.remove(uid)
            # Ensure final flush requests broadcast and
            # kick ranks 1 -> n out of the while loop
            self._bcast_requests(force=True)
        else:
            # Ranks 1 -> n just run generate() until there are no more requests
            while self.scheduled_requests:
                self.generate()

        outputs = [
            r for idx,
            r in sorted(zip(uids_complete_order,
                            outputs),
                        key=lambda pair: pair[0])
        ]

        if self.model_config.all_rank_output:
            outputs = self._bcast_responses(outputs)

        return outputs

    def _put_request(self, uid: int, input: str, kwargs: Dict[str, Any]) -> None:
        self.result_queues[self.tid] = queue.Queue()
        input_tokens = self.tokenizer.encode(input)
        request = self.make_request(self.tid, uid, input_tokens, kwargs)
        self.request_queue.put(request)

    def _get_response(self) -> Tuple[int, Response]:
        result = self.result_queues[self.tid].get()
        uid = result[0]
        generated_tokens = self.tokenizer.decode(result[1])
        response = self.make_response(generated_tokens, result[2], result[3], result[4])
        return uid, response

    def _bcast_responses(self, responses: List[Response]) -> List[Response]:
        if self.is_rank_0:
            data_dicts = [r.to_msg_dict() for r in responses]
            json_data = ujson.dumps(data_dicts)
            self.socket.send_string(json_data)
        else:
            json_data = self.socket.recv_string()
            data_dicts = ujson.loads(json_data)
            responses = [Response.from_msg_dict(msg) for msg in data_dicts]
        return responses

    def destroy(self) -> None:
        del self.inference_engine
        self.socket.close()
        self._zmq_context.term()
        gc.collect()
        get_accelerator().empty_cache()
        self._destroyed = True


class MIIAsyncPipeline(RaggedBatchBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uids = set()
        self.lock = threading.Lock()
        self.thread = None
        self.stop_thread = False
        self._is_shutdown = False
        self.UID_RANGE_LB = 1
        self.UID_RANGE_UB = 10000

    def __call__(self) -> None:
        # CUDA device gets reset, must set it again to avoid problems
        get_accelerator().set_device(int(os.getenv("LOCAL_RANK", "0")))
        while True:
            self.generate()

            if (self.stop_thread and self.request_queue.empty()
                    and all(q.empty() for q in self.result_queues.values())):
                break

    def _get_uid(self) -> int:
        with self.lock:
            uid = random.randrange(self.UID_RANGE_LB, self.UID_RANGE_UB)
            while uid in self.uids:
                uid = random.randrange(self.UID_RANGE_LB, self.UID_RANGE_UB)
            self.uids.add(uid)

        return uid

    def put_request(self, prompt: str, kwargs: Dict) -> int:
        # TODO: We should avoid any request/response work with non-rank 0, but
        # this requires some refactoring how we do the put and request in
        # `ModelResponse`
        #if not self.is_rank_0:
        #    return
        if self.stop_thread:
            raise RuntimeError("The request queue was shutdown.")

        uid = self._get_uid()

        # Temporary hack to avoid non-rank 0 processes not shutting down. See
        # related TODO above.
        if not self.is_rank_0:
            return uid

        tid = threading.get_ident()
        with self.lock:
            if tid not in self.result_queues:
                self.result_queues[tid] = queue.Queue()

        input_tokens = self.tokenizer.encode(prompt)
        request = self.make_request(tid, uid, input_tokens, kwargs)
        self.request_queue.put(request)

        return uid

    def get_response(self) -> Tuple[int, Response]:
        # TODO: We should avoid any request/response work with non-rank 0, but
        # this requires some refactoring how we do the put and request in
        # `ModelResponse`
        if not self.is_rank_0:
            return -1, Response(generated_text="",
                            prompt_length=None,
                            generated_length=None,
                            finish_reason=None)
        tid = threading.get_ident()
        result = self.result_queues[tid].get()
        uid = result[0]
        generated_token_ids = result[1]
        if len(generated_token_ids) == 0:
            generated_text = ""
        else:
            generated_text = self.tokenizer.decode(generated_token_ids)
        response = self.make_response(generated_text, result[2], result[3], result[4])
        return uid, response

    def start(self) -> None:
        self.thread = threading.Thread(target=self, daemon=True)
        self.thread.start()

    def shutdown(self) -> None:
        self.stop_thread = True
        self.thread.join()
        self._is_shutdown = True

    def is_shutdown(self) -> bool:
        return self._is_shutdown

    def flush_uid(self, uid: int) -> None:
        with self.lock:
            if self.is_rank_0:
                self._queue_flush_request(uid)
            self.uids.remove(uid)
