# Propagating C API Changes from bleve-faiss-C

This repo is a pure Go wrapper around the C API exposed by
[bleve-faiss-C](https://github.com/francesco55/bleve-faiss-C).
When that C library gains new functions, three things must happen in sequence:
rebuild the library, install it, then add Go bindings here.

---

## 1. Build and install the updated C library

```bash
# From the bleve-faiss-C local clone
cmake --build build --target faiss_c
sudo cmake --install build
```

The install copies two things to `/usr/local/`:
- the updated shared library (`libfaiss_c.dylib`)
- the updated headers (`include/faiss/c_api/`)

Verify the new symbols are present before moving on:

```bash
nm /usr/local/lib/libfaiss_c.dylib | grep <new_function_name>
```

---

## 2. Add Go bindings

New C API functions follow the conventions below.  All IVF extensions go in
[index_ivf.go](index_ivf.go); other extensions go in [index.go](index.go).

### Pattern for a simple IVF method

```go
func (idx *faissIndex) MyNewMethod(arg int) error {
    ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
    if ivfPtr == nil {
        return ErrNotIVFIndex
    }
    if c := C.faiss_IndexIVF_my_new_function(ivfPtr, C.int(arg)); c != 0 {
        return newFaissError(ErrSetParamsFailed, getLastError(), int(c))
    }
    return nil
}
```

Key rules inherited from the C API conventions:
- Functions that return `int` use `0 = success`, non-zero = error — wrap with `newFaissError`.
- Pure predicate functions (e.g. `has_partition_map` returning 0/1) are **not** error-returning; check `!= 0` directly.
- Output-pointer functions (`get_*`) pass a `*C.T` local variable and read it after the call.
- Array arguments follow `(ptr, size)` — convert Go slices with `(*C.T)(&slice[0])` and `C.size_t(len(slice))`.

### Exposing the method on the Index interface

Add the new signature to the `Index` interface in [index.go](index.go) alongside
related methods.  Keep the comment focused on the *why* (behaviour contract),
not the *what*.

### Error codes

If the new method needs a new sentinel error, add it to the appropriate block
in [errors.go](errors.go):

```go
ErrMyNewCondition = errors.New("short description")
```

---

## 3. Verify

```bash
go build ./...   # must be clean
go vet ./...
```

---

## 4. IDE note — stale gopls cache

After `sudo cmake --install` updates the headers on disk, VS Code's Go language
server (gopls) may still show "undefined: C.faiss_…" errors even though the
build is clean.  Fix:

- **Reload the window**: `Cmd+Shift+P` → *Developer: Reload Window*
- or **Restart the language server**: `Cmd+Shift+P` → *Go: Restart Language Server*
