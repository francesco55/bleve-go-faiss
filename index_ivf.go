package faiss

/*
#include <faiss/c_api/IndexIVFFlat_c.h>
#include <faiss/c_api/MetaIndexes_c.h>
#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/IndexIVF_c.h>
#include <faiss/c_api/IndexIVF_c_ex.h>
#include <faiss/c_api/IndexScalarQuantizer_c.h>
*/
import "C"
import "unsafe"

func (idx *faissIndex) SetDirectMap(mapType int) (err error) {

	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return ErrNotIVFIndex
	}
	if c := C.faiss_IndexIVF_set_direct_map(
		ivfPtr,
		C.int(mapType),
	); c != 0 {
		err = newFaissError(ErrSetParamsFailed, getLastError(), int(c))
	}
	return err
}

func (idx *faissIndex) GetSubIndex() (Index, error) {

	ptr := C.faiss_IndexIDMap2_cast(idx.cPtr())
	if ptr == nil {
		return nil, ErrNotIDMapIndex
	}

	subIdx := C.faiss_IndexIDMap2_sub_index(ptr)
	if subIdx == nil {
		return nil, ErrNotIDMapIndex
	}

	return &IndexImpl{&faissIndex{subIdx}}, nil
}

// pass nprobe to be set as index time option for IVF indexes only.
// varying nprobe impacts recall but with an increase in latency.
func (idx *faissIndex) SetNProbe(nprobe int32) {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return
	}
	C.faiss_IndexIVF_set_nprobe(ivfPtr, C.size_t(nprobe))
}

func (idx *faissIndex) IVFParams() (nprobe, nlist int) {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return 0, 0
	}
	return int(C.faiss_IndexIVF_nprobe(ivfPtr)),
		int(C.faiss_IndexIVF_nlist(ivfPtr))
}

func (idx *faissIndex) IsSQIndex() bool {
	sqPtr := C.faiss_IndexScalarQuantizer_cast(idx.cPtr())
	return sqPtr != nil
}

func (idx *faissIndex) SetQuantizers(srcIndex Index) error {
	if !(idx.IsIVFIndex() && srcIndex.IsIVFIndex()) &&
		!(idx.IsSQIndex() && srcIndex.IsSQIndex()) {
		return ErrSetQuantizerNotSupported
	}
	c := C.faiss_Set_quantizers(idx.idx, srcIndex.cPtr())
	if c != 0 {
		return newFaissError(ErrSetQuantizerFailed, getLastError(), int(c))
	}
	return nil
}

// IVFListSize returns the number of vectors in posting list list_no.
func (idx *faissIndex) IVFListSize(listNo int) (int, error) {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return 0, ErrNotIVFIndex
	}
	return int(C.faiss_IndexIVF_get_list_size(ivfPtr, C.size_t(listNo))), nil
}

// IVFCodeSize returns the number of bytes per stored code/vector in the index.
// For IndexIVFFlat this equals d * 4 (raw float32 vectors).
func (idx *faissIndex) IVFCodeSize() (int, error) {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return 0, ErrNotIVFIndex
	}
	return int(C.faiss_IndexIVF_code_size(ivfPtr)), nil
}

// IVFListIDs returns the vector IDs stored in posting list list_no.
func (idx *faissIndex) IVFListIDs(listNo int) ([]int64, error) {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return nil, ErrNotIVFIndex
	}
	listSize := int(C.faiss_IndexIVF_get_list_size(ivfPtr, C.size_t(listNo)))
	if listSize == 0 {
		return nil, nil
	}
	buf := make([]int64, listSize)
	C.faiss_IndexIVF_invlists_get_ids(
		ivfPtr,
		C.size_t(listNo),
		(*C.idx_t)(unsafe.Pointer(&buf[0])),
	)
	return buf, nil
}

// IVFListCodes returns the raw byte codes for all vectors in posting list
// list_no. Each code is IVFCodeSize() bytes. For IndexIVFFlat, reinterpret
// each code as []float32 of length d.
func (idx *faissIndex) IVFListCodes(listNo int) ([]byte, error) {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return nil, ErrNotIVFIndex
	}
	listSize := int(C.faiss_IndexIVF_get_list_size(ivfPtr, C.size_t(listNo)))
	codeSize := int(C.faiss_IndexIVF_code_size(ivfPtr))
	if listSize == 0 {
		return nil, nil
	}
	buf := make([]byte, listSize*codeSize)
	C.faiss_IndexIVF_invlists_get_codes(
		ivfPtr,
		C.size_t(listNo),
		(*C.uint8_t)(unsafe.Pointer(&buf[0])),
	)
	return buf, nil
}
