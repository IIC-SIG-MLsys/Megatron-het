# Moe support
version suggest : > v0.12.0

- megatron/core/transformer/moe/moe_layer.py

    ```
    MoEAllGatherTokenDispatcher, <
    MoEAlltoAllTokenDispatcher, <
    MoEFlexTokenDispatcher, (depend on DeepEP)
    ```
    we should support these types of TokenDispatcher

- Each TokenDisptcher has a `token_dispatch()` and `token_compine()`

    ```
    gather_from_sequence_parallel_region, <
    all_to_all, <
    reduce_scatter_to_sequence_parallel_region,
    fused_dispatch (depend on DeepEP)
    ```

- We should implement

    gather_from_sequence_parallel_region
    ```
    torch.distributed._reduce_scatter_base
    torch.distributed._all_gather_base
    ```
    alltoall needs:
    ```
    torch.distributed.all_to_all_single
    ```
