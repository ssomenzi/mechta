// Reference for step 2 of M1: decodes the first N values of a GGUF tensor
// with the ggml function itself (to_float), and writes them as float32.
// Usage: deq_ref <file.gguf> <tensor name> <N> <output.f32>
#include "ggml.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: deq_ref file.gguf tensor N output.f32\n"); return 1; }
    struct gguf_init_params ip = { .no_alloc = true, .ctx = NULL };
    struct gguf_context *g = gguf_init_from_file(argv[1], ip);
    if (!g) { fprintf(stderr, "gguf_init failed\n"); return 1; }
    int64_t id = gguf_find_tensor(g, argv[2]);
    if (id < 0) { fprintf(stderr, "tensor not found: %s\n", argv[2]); return 1; }
    enum ggml_type t = gguf_get_tensor_type(g, id);
    int64_t n = atoll(argv[3]);
    size_t nbytes = ggml_row_size(t, n);
    size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
    FILE *f = fopen(argv[1], "rb"); fseeko(f, off, SEEK_SET);
    void *raw = malloc(nbytes); if (fread(raw, 1, nbytes, f) != nbytes) { fprintf(stderr, "short read\n"); return 1; } fclose(f);
    float *out = malloc(n * sizeof(float));
    ggml_get_type_traits(t)->to_float(raw, out, n);
    FILE *o = fopen(argv[4], "wb"); fwrite(out, sizeof(float), n, o); fclose(o);
    printf("%s: type %s, %lld values, %zu bytes from offset %zu\n", argv[2], ggml_type_name(t), (long long) n, nbytes, off);
    return 0;
}
