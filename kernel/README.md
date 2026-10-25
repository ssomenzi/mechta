# Kernel: TTM swapout fix for the Radeon 780M

## The problem

On the Radeon 780M, some Linux 7.1/7.2 kernels crash when TTM swaps GPU buffers out of memory under pressure
(for example when a large model fills the VRAM carve-out and spills into GTT). After a swapout the buffer is
left on the LRU bulk move, and the bulk move is later used after it has been freed (use-after-free).

## The fix

The upstream fix (dri-devel, applied by Christian König to drm-misc-fixes) changes one condition in
`ttm_bo_swapout_cb()` in `drivers/gpu/drm/ttm/ttm_bo.c`:

```diff
-		if (!ret) {
+		if (ret > 0) {
```

so the resource is removed from the bulk move only when the swapout actually happened. One linux-next commit
(`3db7d7d58341`) had applied it to the wrong line; the patch here is the intended one. The fix is included in
the stable 7.2.8 kernel (`fcfe64715b42`).

`patch-upstream/omarchy-0361.patch` is that fix as packaged by a distribution (Omarchy, commit `f5c9441888`,
patch `0361-drm-ttm-swapped-out-resource-leaves-bulk-move.patch`).

## How it was tested

A 7.2.5 kernel built with this single extra patch (same configuration as the stock one) was run on the test
machine (a Radeon 780M mini PC):

- normal inference with the VRAM carve-out full and the model spilling into GTT, the condition that crashed
  before: no crash and no TTM, `list_del corruption`, Oops or GPU reset messages in the kernel log;
- a memory-pressure test with forced swapouts (TTM shrink and a lowered `pages_limit` during model loads):
  the swapout path was exercised (several GB moved to shmem) with no kernel crash, no amdgpu errors and no
  GPU reset. In one run with the 16 GB carve-out at 99.6%, the model produced invalid output after reloads;
  this did not reproduce after a reboot with a smaller carve-out, and its cause was not determined.

A distribution 7.2.8 kernel that contains the fix also ran inference without faults.
