from __future__ import annotations

import torch

from tinyvllm.kernels.kv_store import store_kv
from tinyvllm.kernels.paged_decode import paged_decode_attention
from tinyvllm.paged_kv import PagedKVCache
from tinyvllm.runtime.contracts import ModelInput
from tinyvllm.runtime.inputs import DecodeInputs


class _DecodeAttention:
    """Static single-query operator sequence, without CPU batch dispatch."""

    def __init__(self, inputs: ModelInput, cache: PagedKVCache) -> None:
        self.kv = inputs.kv
        self.cache = cache
        
        
    def __call__(self, layer_idx, q, k, v, *, scale):
        store_kv(self.cache, layer_idx, self.kv.slot_mapping, k, v)
        
        output = torch.empty_like(q)
        paged_decode_attention(
            q,
            self.cache,
            layer_idx,
            self.kv.decode,
            output,
            scale=scale,
        )

        return output
    
    
class DecodeGraphs:
    """Startup-only capture; Runner keeps model, cache and inputs alive.
    
    Initialize before serving: no other thread in this process may execute
    CUDA work during capture. One Runner owns all subsequent graph replays.
    """

    def __init__(self, stream: torch.cuda.Stream) -> None:
        self.stream = stream
        self.entries: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        
        
    @torch.inference_mode()
    def capture(
        self,
        model: torch.nn.Module,
        inputs: DecodeInputs,
        cache: PagedKVCache,
    ) -> None:
        limit = min(inputs.capacity, cache.num_blocks, 16)

        sizes = tuple(
            size
            for size in (1, 2, 4, 8, 12, 16)
            if size < limit
        ) + (limit,)

        count = limit
        
        # No real requests exist yet. Each dummy row owns one distinct block.
        tokens, positions, slots, offsets, lengths, blocks = inputs.device_views
        
        tokens[:count].zero_()
        positions[:count].zero_()
        slots[:count].copy_(inputs.rows[:count] * cache.block_size)
        offsets[:count + 1].copy_(
            torch.arange(
                count + 1,
                device=inputs.device,
                dtype=torch.long,
            ),
        )
        lengths[:count].fill_(1)
        blocks[:count].copy_(inputs.rows[:count])

        capture_stream = torch.cuda.Stream(device=inputs.device)
        capture_stream.wait_stream(self.stream)
        
        try:
            with torch.cuda.stream(capture_stream):
                for size in reversed(sizes):
                    model_input = inputs.view(size)
                    attention = _DecodeAttention(model_input, cache)

                    def forward():
                        hidden = model(
                            model_input.input_ids,
                            model_input.positions,
                            attention=attention,
                        )

                        # All rows are sampled; no index_select is needed.
                        return model.compute_logits(hidden)
                    
                    for _ in range(3):
                        forward()
                        
                    capture_stream.synchronize()
                    
                    graph = torch.cuda.CUDAGraph()
                    
                    # Independent pools allow arbitrary batch-size replay order.
                    with torch.cuda.graph(graph, stream=capture_stream):
                        logits = forward()
                        
                    self.entries[size] = (graph, logits)

        except Exception as cause:
            try:
                capture_stream.synchronize()
            except Exception as cleanup_error:
                cause.add_note(f"capture stream cleanup: {cleanup_error!r}")

            raise
        else:
            # Startup joins the side stream before serving.
            capture_stream.synchronize()
            
            # Search nearest upper size without sorting on the decode hot path.
            self.entries = dict(sorted(self.entries.items()))
            
            
    def replay(self, count: int) -> torch.Tensor | None:
        for size, (graph, logits) in self.entries.items():
            if count <= size:
                graph.replay()
                return logits[:count]

        return None
    
    
    def close(self) -> None:
        # Graph outputs/private pools must outlive all queued replays.
        self.stream.synchronize()
        self.entries.clear()