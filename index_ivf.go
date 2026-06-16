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
import (
	"encoding/json"
	"math"
	"unsafe"
)

// RemoteProbe is a centroid probe that must be forwarded to another worker
// because that worker owns the corresponding inverted list.
type RemoteProbe struct {
	WorkerID   int
	CentroidID int64
	Distance   float32
}

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

// InitPartitionMap allocates a partition map on the index and sets this node's
// worker ID.  Must be called before SetListWorker or SearchLocalShard.
func (idx *faissIndex) InitPartitionMap(myWorkerID int) error {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return ErrNotIVFIndex
	}
	if c := C.faiss_IndexIVF_init_partition_map(ivfPtr, C.int(myWorkerID)); c != 0 {
		return newFaissError(ErrSetParamsFailed, getLastError(), int(c))
	}
	return nil
}

// SetListWorker assigns inverted list listNo to the given workerID in the
// partition map.
func (idx *faissIndex) SetListWorker(listNo int, workerID int) error {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return ErrNotIVFIndex
	}
	if c := C.faiss_IndexIVF_set_list_worker(ivfPtr, C.size_t(listNo), C.int(workerID)); c != 0 {
		return newFaissError(ErrSetParamsFailed, getLastError(), int(c))
	}
	return nil
}

// GetListWorker returns the worker ID that owns inverted list listNo.
func (idx *faissIndex) GetListWorker(listNo int) (int, error) {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return 0, ErrNotIVFIndex
	}
	var workerID C.int
	if c := C.faiss_IndexIVF_get_list_worker(ivfPtr, C.size_t(listNo), &workerID); c != 0 {
		return 0, newFaissError(ErrInspectIndexFailed, getLastError(), int(c))
	}
	return int(workerID), nil
}

// HasPartitionMap returns true if a partition map has been initialised on this index.
func (idx *faissIndex) HasPartitionMap() bool {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return false
	}
	return C.faiss_IndexIVF_has_partition_map(ivfPtr) != 0
}

// CopyListsTo copies the inverted lists identified by listNos from this index
// into dst.  Both indexes must have identical nlist and code_size.
func (idx *faissIndex) CopyListsTo(dst Index, listNos []int64) error {
	srcPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if srcPtr == nil {
		return ErrNotIVFIndex
	}
	dstPtr := C.faiss_IndexIVF_cast(dst.cPtr())
	if dstPtr == nil {
		return ErrNotIVFIndex
	}
	if len(listNos) == 0 {
		return nil
	}
	if c := C.faiss_IndexIVF_copy_lists_to(
		srcPtr,
		dstPtr,
		(*C.idx_t)(&listNos[0]),
		C.size_t(len(listNos)),
	); c != 0 {
		return newFaissError(ErrInspectIndexFailed, getLastError(), int(c))
	}
	return nil
}

// SearchLocalShard performs coarse quantization using the index's current nprobe
// setting, routes each centroid probe to local or remote based on the partition
// map, searches the local lists, and returns the partial results together with
// probes that the caller must forward to remote workers.
//
// myWorkerID must match the ID passed to InitPartitionMap.
func (idx *faissIndex) SearchLocalShard(
	x []float32, k int64, sel Selector, params json.RawMessage, myWorkerID int,
) ([]float32, []int64, []RemoteProbe, error) {
	if !idx.IsIVFIndex() {
		return nil, nil, nil, ErrNotIVFIndex
	}
	// Partition map must be initialised; without it we cannot decide which
	// centroids are local and which must be forwarded.
	if !idx.HasPartitionMap() {
		return nil, nil, nil, ErrNoPartitionMap
	}

	// Step 1 — coarse quantization.
	// Ask the quantizer for the nprobe closest centroids and their distances.
	// nprobe is read from the index so the caller controls recall vs. latency
	// via SetNProbe, the same knob used for regular Search calls.
	nprobe, _ := idx.IVFParams()
	centroidIDs, centroidDis, err := idx.ObtainClustersWithDistancesFromIVFIndex(x, nil, int64(nprobe))
	if err != nil {
		return nil, nil, nil, err
	}

	// Step 2 — partition probes into local and remote.
	// For each centroid returned by coarse quantization, look up its owner in
	// the partition map.  Centroids owned by this worker are searched locally;
	// all others are packaged as RemoteProbes for the caller to forward.
	var localIDs []int64
	var localDis []float32
	var remoteProbes []RemoteProbe

	for i, cid := range centroidIDs {
		// Negative centroid IDs are FAISS sentinels meaning "no result" (the
		// quantizer found fewer clusters than nprobe); skip them.
		if cid < 0 {
			continue
		}
		wid, err := idx.GetListWorker(int(cid))
		if err != nil {
			return nil, nil, nil, err
		}
		if wid == myWorkerID {
			localIDs = append(localIDs, cid)
			localDis = append(localDis, centroidDis[i])
		} else {
			remoteProbes = append(remoteProbes, RemoteProbe{
				WorkerID:   wid,
				CentroidID: cid,
				Distance:   centroidDis[i],
			})
		}
	}

	// Step 3 — search local lists (or return empty results if there are none).
	// When no centroid is local we still return properly-shaped distance and
	// label slices filled with FAISS sentinel values (-1 / MaxFloat32) so the
	// caller can top-k merge this shard's output uniformly with remote results.
	n := len(x) / idx.D()
	if len(localIDs) == 0 {
		distances := make([]float32, int64(n)*k)
		labels := make([]int64, int64(n)*k)
		for i := range labels {
			labels[i] = -1
			distances[i] = math.MaxFloat32
		}
		return distances, labels, remoteProbes, nil
	}

	// centroidsToProbe = len(localIDs) so every local list is scanned; the
	// effective probe count is further capped inside SearchClustersFromIVFIndex
	// by any nprobe override carried in params.
	distances, labels, err := idx.SearchClustersFromIVFIndex(
		localIDs, localDis, len(localIDs), x, k, sel, params,
	)
	if err != nil {
		return nil, nil, nil, err
	}
	return distances, labels, remoteProbes, nil
}
