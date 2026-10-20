from __future__ import annotations

from collections.abc import Callable, Sequence
from hashlib import sha256
from typing import TYPE_CHECKING

import struct


if TYPE_CHECKING:
    from tinyvllm.cache import BlockPool, BlockRef
    
    
class PrefixCache:
    """Token-prefix policy: BlockManager applies ownership changes."""

    def __init__(
        self,
        *,
        block_size: int,
        pool: BlockPool,
        enabled: bool,
    ) -> None:
        self.block_size = block_size
        self.pool = pool
        self.enabled = enabled
        
        self._request_keys: dict[int, list[bytes]] = {}
        self._published_blocks: dict[int, int] = {}
        
        self.hits = 0
        self.hit_tokens = 0
        
    
    @staticmethod
    def _hash_block(parent: bytes, tokens: Sequence[int]) -> bytes:
        token_bytes = struct.pack(f"{len(tokens)}i", *tokens)
        return sha256(parent + token_bytes).digest()
    
    
    def match(
        self,
        request_id: int,
        token_ids: Sequence[int],
    ) -> tuple[BlockRef, ...]:
        if not self.enabled:
            return ()
        
        keys = self._request_keys.get(request_id)
        
        if keys is None:
            keys = []
            parent = b""
            full_end = len(token_ids) // self.block_size * self.block_size
            
            for start in range(0, full_end, self.block_size):
                parent = self._hash_block(
                    parent,
                    token_ids[start:start + self.block_size],
                )
                keys.append(parent)
                
            self._request_keys[request_id] = keys
            
        # Leave a query for logits; a block-aligned prompt recomputes its tail block.
        limit = (len(token_ids) - 1) // self.block_size
        refs = []
        
        for key in keys[:limit]:
            ref = self.pool.lookup(key)
            
            if ref is None:
                break
            
            refs.append(ref)
            
        return tuple(refs)

        
    def next_prefill_key(
        self,
        request_id: int,
        token_start: int,
        prompt_len: int,
        chunk_size: int,
    ) -> bytes | None:
        """Identify the reusable prefix that one producer should compute next."""
        first_block = token_start // self.block_size
        reusable_blocks = (prompt_len - 1) // self.block_size

        if not self.enabled or first_block >= reusable_blocks:
            return None

        span_blocks = (
            chunk_size + self.block_size - 1
        ) // self.block_size

        block_index = min(
            first_block + span_blocks,
            reusable_blocks,
        ) - 1

        return self._request_keys[request_id][block_index]
    
    
    def validate_match(
        self,
        request_id: int,
        refs: tuple[BlockRef, ...],
    ) -> None:
        keys = self._request_keys.get(request_id, ())
        
        if len(keys) < len(refs):
            raise ValueError(
                "prefix attachment requires a matched new request"
            )

        for key, ref in zip(keys, refs):
            if self.pool.lookup(key) != ref:
                raise ValueError("prefix match is no longer cached")
            
            
    def publish(
        self,
        request_id: int,
        table: tuple[BlockRef, ...],
        valid_tokens: int,
        token_slice: Callable[[int, int], Sequence[int]],
    ) -> tuple[tuple[int, BlockRef], ...]:
        """Index newly computed full blocks and return canonical replacements."""
        if not self.enabled:
            return ()
        
        full_count = valid_tokens // self.block_size
        published = self._published_blocks.get(request_id, 0)
        
        if full_count == published:
            return ()
        
        keys = self._request_keys.setdefault(request_id, [])
        replacements = []
        
        for index in range(published, full_count):
            if index == len(keys):
                start = index * self.block_size
                tokens = token_slice(start, start + self.block_size)
                
                if len(tokens) != self.block_size:
                    raise ValueError(
                        "computed prefix must contain a complete block"
                    )

                keys.append(
                    self._hash_block(
                        keys[-1] if keys else b"",
                        tokens,
                    )
                )

            ref = table[index]
            canonical = self.pool.publish(keys[index], ref)
            
            if canonical != ref:
                replacements.append((index, canonical))
                
        self._published_blocks[request_id] = full_count
        return tuple(replacements)

        
    def commit_matches(
        self,
        prefixes: dict[int, tuple[BlockRef, ...]],
    ) -> None:
        for request_id, refs in prefixes.items():
            self._published_blocks[request_id] = len(refs)
            
        self.hits += len(prefixes)
        self.hit_tokens += sum(
            len(refs) * self.block_size
            for refs in prefixes.values()
        )

        
    def forget_request(self, request_id: int) -> None:
        self._request_keys.pop(request_id, None)
        self._published_blocks.pop(request_id, None)
        
        
    def clear(self) -> None:
        self.pool.clear_cache()
        self._request_keys.clear()
        self._published_blocks.clear()
        self.hits = 0
        self.hit_tokens = 0