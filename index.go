package faiss

/*
#include <stdlib.h>
#include <faiss/c_api/Index_c.h>
#include <faiss/c_api/IndexIVF_c.h>
#include <faiss/c_api/IndexIVF_c_ex.h>
#include <faiss/c_api/Index_c_ex.h>
#include <faiss/c_api/impl/AuxIndexStructures_c.h>
#include <faiss/c_api/index_factory_c.h>
#include <faiss/c_api/MetaIndexes_c.h>
*/
import "C"
import (
	"encoding/json"
	"math"
	"reflect"
	"sort"
	"unsafe"
)

var reflectStaticSizeFaissIndex uint64

func init() {
	var f faissIndex
	reflectStaticSizeFaissIndex = uint64(reflect.TypeOf(f).Size())
}

// Index is a Faiss index.
//
// Note that some index implementations do not support all methods.
// Check the Faiss wiki to see what operations an index supports.
type Index interface {
	// D returns the dimension of the indexed vectors.
	D() int

	// IsTrained returns true if the index has been trained or does not require
	// training.
	IsTrained() bool

	// Ntotal returns the number of indexed vectors.
	Ntotal() int64

	// set the direct map type for IVF indexes.
	// 0 for No Map
	// 1 for Array
	// 2 for Hash
	SetDirectMap(maptype int) error

	// set the number of probes for IVF indexes
	SetNProbe(nprobe int32)

	// MetricType returns the metric type of the index.
	MetricType() int

	// Train trains the index on a representative set of vectors.
	Train(x []float32) error

	// Add adds vectors to the index.
	Add(x []float32) error

	// AddWithIDs is like Add, but stores xids instead of sequential IDs.
	AddWithIDs(x []float32, xids []int64) error

	// Returns true if the index is an IVF index.
	IsIVFIndex() bool

	// Returns true if the index is a scalar quantization (SQ) index.
	IsSQIndex() bool

	// Returns true if the index has RaBitQ
	HasRaBitQ() bool

	// Returns the IVF parameters nprobe and nlist for IVF indexes.
	IVFParams() (nprobe, nlist int)

	// Applicable only to IVF indexes: Returns a slice where each index represents
	// a cluster (list) ID and the value is the count of selected vectors belonging
	// to that cluster. Only vectors specified by the given Selector are considered.
	ObtainClusterVectorCountsFromIVFIndex(include Selector, nlist int) ([]int64, error)

	// Applicable only to IVF indexes: Returns the centroid IDs in the selector in
	// decreasing order of proximity to query 'x' and their distance from 'x'
	ObtainClustersWithDistancesFromIVFIndex(x []float32, centroids Selector, numCentroids int64) (
		[]int64, []float32, error)

	// Applicable only to IVF indexes: Returns the top k centroid cardinalities and
	// their vectors in chosen order (descending or ascending)
	ObtainKCentroidCardinalitiesFromIVFIndex(limit int, descending bool) ([]uint64, [][]float32, error)

	// fetch centroid count
	Nlist() int

	// Search queries the index with the vectors in x.
	// Returns the IDs of the k nearest neighbors for each query vector and the
	// corresponding distances.
	Search(x []float32, k int64) (distances []float32, labels []int64, err error)

	// SearchWithOptions performs a search with additional optional constraints.
	// - Selector can be used to restrict the search to a subset of the indexed vectors based on their IDs.
	// - params is a JSON object that can contain additional search parameters specific to the index type, such as IVF search parameters.
	SearchWithOptions(x []float32, k int64, sel Selector, params json.RawMessage) (distances []float32, labels []int64, err error)

	// Applicable only to IVF indexes: Search clusters whose IDs are in eligibleCentroidIDs.
	// eligibleCentroidIDs[i] / centroidDis[i] hold the pre-assigned centroid IDs and
	// distances for query i; rows may have different lengths.
	// len(eligibleCentroidIDs) must equal len(x)/D().
	// Returns distances[i] and labels[i] — the k nearest neighbours for query i.
	SearchClustersFromIVFIndex(eligibleCentroidIDs [][]int64, centroidDis [][]float32,
		x []float32, k int64, include Selector, params json.RawMessage) ([][]float32, [][]int64, error)

	Reconstruct(key int64) ([]float32, error)

	ReconstructBatch(keys []int64, recons []float32) ([]float32, error)

	MergeFrom(other Index, add_id int64) error

	// RangeSearch queries the index with the vectors in x.
	// Returns all vectors with distance < radius.
	RangeSearch(x []float32, radius float32) (*RangeSearchResult, error)

	// DistCompute computes the distance between the query vector and the vectors specified by ids.
	DistCompute(x []float32, labels []int64) ([]float32, error)

	// Reset removes all vectors from the index.
	Reset() error

	// RemoveIDs removes the vectors specified by sel from the index.
	// Returns the number of elements removed and error.
	RemoveIDs(sel *IDSelector) (int, error)

	// Close frees the memory used by the index.
	Close()

	// Size estimates the memory footprint of the index in bytes,
	// if the underlying faiss index is memory-mapped and not fully loaded into memory.
	Size() uint64

	// cPtr returns a pointer to the underlying C index struct.
	cPtr() *C.FaissIndex

	// set the quantizers from a source index into this index, applicable only
	// for IVF indexes
	SetQuantizers(source Index) error

	// CodeSize returns the size of the produced codes in bytes.
	CodeSize() (uint64, error)

	IVFListSize(listNo int) (int, error)
	IVFCodeSize() (int, error)
	IVFListIDs(listNo int) ([]int64, error)
	IVFListCodes(listNo int) ([]byte, error)

	// Distributed IVF partition map — set up which worker owns each inverted list.
	InitPartitionMap(myWorkerID int) error
	SetListWorker(listNo int, workerID int) error
	GetListWorker(listNo int) (int, error)
	HasPartitionMap() bool
	// Batched variants: one cgo call for the full list instead of one per entry.
	InitPartitionMapWithOwners(myWorkerID int, listToWorker []int) error
	SetListWorkers(listNos []int64, workerIDs []int) error
	GetListWorkers(listNos []int64) ([]int, error)
	// CopyListsTo copies the inverted lists identified by listNos from this index
	// into dst.  Both indexes must have identical nlist and code_size.
	CopyListsTo(dst Index, listNos []int64) error

	// SearchLocalShard performs coarse quantization, routes centroid probes to
	// local vs. remote workers, searches local lists, and returns the partial
	// results together with probes that must be forwarded to remote workers.
	// myWorkerID must match the ID passed to InitPartitionMap.
	SearchLocalShard(x []float32, k int64, sel Selector, params json.RawMessage, myWorkerID int) (
		distances []float32, labels []int64, remoteProbes []RemoteProbe, err error)

	// ObtainClustersWithWorkersDistancesFromIVFIndex coarse-quantizes the queries
	// in x (len(x)/d queries), looks up the owning worker for each of the nprobe
	// nearest centroids via the partition map, and returns the results grouped by
	// worker per query. Requires a partition map to be initialised on the index.
	ObtainClustersWithWorkersDistancesFromIVFIndex(x []float32, nprobe int64) ([][]WorkerProbes, error)
}

type faissIndex struct {
	idx *C.FaissIndex
}

func (idx *faissIndex) cPtr() *C.FaissIndex {
	return idx.idx
}

func (idx *faissIndex) Size() uint64 {
	rv := reflectStaticSizeFaissIndex
	var size C.size_t
	if code := C.faiss_Index_size(idx.idx, &size); code == 0 {
		rv += uint64(size)
	}
	return rv
}

func (idx *faissIndex) D() int {
	return int(C.faiss_Index_d(idx.idx))
}

func (idx *faissIndex) CodeSize() (uint64, error) {
	var size C.size_t
	if c := C.faiss_Index_sa_code_size(idx.idx, &size); c != 0 {
		return 0, newFaissError(ErrInspectIndexFailed, getLastError(), int(c))
	}
	return uint64(size), nil
}

func (idx *faissIndex) IsTrained() bool {
	return C.faiss_Index_is_trained(idx.idx) != 0
}

func (idx *faissIndex) Ntotal() int64 {
	return int64(C.faiss_Index_ntotal(idx.idx))
}

func (idx *faissIndex) MetricType() int {
	return int(C.faiss_Index_metric_type(idx.idx))
}

func (idx *faissIndex) Train(x []float32) error {
	n := len(x) / idx.D()
	if c := C.faiss_Index_train(idx.idx, C.idx_t(n), (*C.float)(&x[0])); c != 0 {
		return newFaissError(ErrTrainFailed, getLastError(), int(c))
	}
	return nil
}

func (idx *faissIndex) Add(x []float32) error {
	n := len(x) / idx.D()
	if c := C.faiss_Index_add(idx.idx, C.idx_t(n), (*C.float)(&x[0])); c != 0 {
		return newFaissError(ErrAddFailed, getLastError(), int(c))
	}
	return nil
}

func (idx *faissIndex) ObtainClusterVectorCountsFromIVFIndex(includedVectors Selector, nlist int) ([]int64, error) {
	// Applicable only to IVF indexes
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return nil, ErrNotIVFIndex
	}
	// Creating a slice to hold the count of vectors per cluster
	// Since we have nlist clusters, we create a slice of size nlist
	// listCount[i] will hold the count of vectors in cluster i
	listCount := make([]int64, nlist)
	// Creating a FAISS selector based on the include bitmap.
	params, err := NewStandardSearchParams(includedVectors)
	if err != nil {
		return nil, err
	}
	defer params.Delete()
	// Calling the C function to populate listCount
	// with the count of vectors per cluster, considering only
	// the vectors specified in the include selector.
	if c := C.faiss_IndexIVF_list_vector_count(
		ivfPtr,
		(*C.idx_t)(unsafe.Pointer(&listCount[0])),
		C.size_t(nlist),
		params.sp,
	); c != 0 {
		return nil, newFaissError(ErrInspectIndexFailed, getLastError(), int(c))
	}
	return listCount, nil
}

func (idx *faissIndex) IsIVFIndex() bool {
	if ivfIdx := C.faiss_IndexIVF_cast(idx.cPtr()); ivfIdx == nil {
		return false
	}
	return true
}

func (idx *faissIndex) HasRaBitQ() bool {
	return C.faiss_IndexIVF_has_RaBitQ(idx.idx) == 0
}

func (idx *faissIndex) ObtainClustersWithDistancesFromIVFIndex(x []float32, includedCentroids Selector, numCentroids int64) (
	[]int64, []float32, error) {
	// Applicable only to IVF indexes
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return nil, nil, ErrNotIVFIndex
	}
	params, err := NewStandardSearchParams(includedCentroids)
	if err != nil {
		return nil, nil, err
	}
	defer params.Delete()

	n := len(x) / idx.D()
	if n == 0 {
		return nil, nil, nil
	}

	// Output layout is row-major: [query 0 results ... query n-1 results],
	// each row has numCentroids entries.
	centroids := make([]int64, int64(n)*numCentroids)
	centroidDistances := make([]float32, int64(n)*numCentroids)

	if c := C.faiss_IndexIVF_search_closest_eligible_centroids(
		ivfPtr,
		(C.idx_t)(n),
		(*C.float)(&x[0]),
		(C.idx_t)(numCentroids),
		(*C.float)(&centroidDistances[0]),
		(*C.idx_t)(&centroids[0]),
		params.sp,
	); c != 0 {
		return nil, nil, newFaissError(ErrSearchFailed, getLastError(), int(c))
	}

	return centroids, centroidDistances, nil
}

func (idx *faissIndex) ObtainKCentroidCardinalitiesFromIVFIndex(limit int, descending bool) (
	[]uint64, [][]float32, error) {
	if limit <= 0 {
		return nil, nil, nil
	}

	nlist := int(C.faiss_IndexIVF_nlist(idx.idx))
	if nlist == 0 {
		return nil, nil, nil
	}

	centroidCardinalities := make([]C.size_t, nlist)

	// Allocate a flat buffer for all centroids, then slice it per centroid
	d := idx.D()
	flatCentroids := make([]float32, nlist*d)

	// Call the C function to fill centroid vectors and cardinalities
	c := C.faiss_IndexIVF_get_centroids_and_cardinality(
		idx.idx,
		(*C.float)(&flatCentroids[0]),
		(*C.size_t)(&centroidCardinalities[0]),
		nil,
	)
	if c != 0 {
		return nil, nil, newFaissError(ErrInspectIndexFailed, getLastError(), int(c))
	}

	topIndices := getIndicesOfKCentroidCardinalities(
		centroidCardinalities,
		min(limit, nlist),
		descending)

	rvCardinalities := make([]uint64, len(topIndices))
	rvCentroids := make([][]float32, len(topIndices))

	for i, idx := range topIndices {
		rvCardinalities[i] = uint64(centroidCardinalities[idx])
		rvCentroids[i] = flatCentroids[idx*d : (idx+1)*d]
	}

	return rvCardinalities, rvCentroids, nil

}

func getIndicesOfKCentroidCardinalities(cardinalities []C.size_t, k int, descending bool) []int {
	n := len(cardinalities)
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	// Sort only the indices based on cardinality values
	sort.Slice(indices, func(i, j int) bool {
		if descending {
			return cardinalities[indices[i]] > cardinalities[indices[j]]
		}
		return cardinalities[indices[i]] < cardinalities[indices[j]]
	})
	if k >= n {
		return indices
	}

	return indices[:k]
}
func (idx *faissIndex) Nlist() int {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return 0
	}
	return int(C.faiss_IndexIVF_nlist(idx.idx))
}

func (idx *faissIndex) SearchClustersFromIVFIndex(eligibleCentroidIDs [][]int64, centroidDis [][]float32,
	x []float32, k int64, include Selector, params json.RawMessage) ([][]float32, [][]int64, error) {
	ivfPtr := C.faiss_IndexIVF_cast(idx.cPtr())
	if ivfPtr == nil {
		return nil, nil, ErrNotIVFIndex
	}
	if include == nil {
		return nil, nil, ErrSelectorNil
	}

	n := len(x) / idx.D()
	if n == 0 {
		return nil, nil, nil
	}
	if len(eligibleCentroidIDs) != n || len(centroidDis) != n {
		return nil, nil, ErrInvalidArgument
	}

	// Find the widest row; this becomes the FAISS assign-array stride (nprobe).
	// Shorter rows are padded with the FAISS sentinel -1, which
	// search_preassigned skips automatically.
	maxProbe := 0
	for _, row := range eligibleCentroidIDs {
		if len(row) > maxProbe {
			maxProbe = len(row)
		}
	}
	if maxProbe == 0 {
		distances := make([][]float32, n)
		labels := make([][]int64, n)
		for i := range labels {
			distances[i] = make([]float32, k)
			labels[i] = make([]int64, k)
			for j := range labels[i] {
				labels[i][j] = -1
				distances[i][j] = math.MaxFloat32
			}
		}
		return distances, labels, nil
	}

	// Build flat n*maxProbe arrays in row-major order, padding short rows with -1.
	flatIDs := make([]int64, n*maxProbe)
	flatDis := make([]float32, n*maxProbe)
	for i := range flatIDs {
		flatIDs[i] = -1
	}
	for q := 0; q < n; q++ {
		base := q * maxProbe
		copy(flatIDs[base:], eligibleCentroidIDs[q])
		copy(flatDis[base:], centroidDis[q])
	}

	tempParams := &defaultSearchParamsIVF{
		// Nlist is the per-query centroid count so ivf_nprobe_pct scales against
		// the pre-assigned set, not the full index nlist.
		Nlist:  maxProbe,
		Nprobe: maxProbe,
	}
	searchParams, err := NewSearchParams(idx, params, include, tempParams)
	if err != nil {
		return nil, nil, err
	}
	defer searchParams.Delete()

	// A JSON params override may reduce nprobe below maxProbe.  FAISS strides
	// the assign array by nprobe, so compact rows when that happens so query i
	// reads from the correct flat-array offset.
	effectiveNprobe := int(min(getNProbeFromSearchParams(searchParams), int32(maxProbe)))
	if effectiveNprobe < maxProbe {
		compactIDs := make([]int64, n*effectiveNprobe)
		compactDis := make([]float32, n*effectiveNprobe)
		for q := 0; q < n; q++ {
			src := q * maxProbe
			dst := q * effectiveNprobe
			copy(compactIDs[dst:], flatIDs[src:src+effectiveNprobe])
			copy(compactDis[dst:], flatDis[src:src+effectiveNprobe])
		}
		flatIDs = compactIDs
		flatDis = compactDis
	}

	flatDistances := make([]float32, int64(n)*k)
	flatLabels := make([]int64, int64(n)*k)

	if c := C.faiss_IndexIVF_search_preassigned_with_params(
		ivfPtr,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		(*C.idx_t)(&flatIDs[0]),
		(*C.float)(&flatDis[0]),
		(*C.float)(&flatDistances[0]),
		(*C.idx_t)(&flatLabels[0]),
		C.int(0),
		searchParams.sp,
	); c != 0 {
		return nil, nil, newFaissError(ErrSearchFailed, getLastError(), int(c))
	}

	// Slice the flat results into per-query rows.
	distances := make([][]float32, n)
	labels := make([][]int64, n)
	for q := 0; q < n; q++ {
		distances[q] = flatDistances[int64(q)*k : int64(q+1)*k]
		labels[q] = flatLabels[int64(q)*k : int64(q+1)*k]
	}
	return distances, labels, nil
}

func (idx *faissIndex) AddWithIDs(x []float32, xids []int64) error {
	n := len(x) / idx.D()
	if c := C.faiss_Index_add_with_ids(
		idx.idx,
		C.idx_t(n),
		(*C.float)(&x[0]),
		(*C.idx_t)(&xids[0]),
	); c != 0 {
		return newFaissError(ErrAddFailed, getLastError(), int(c))
	}
	return nil
}

// Always use SearchWithOptions for indexes involving RaBitQ, as
// simple Search is highly unoptimized for RaBitQ indexes and
// will not leverage the quantizer for search.
func (idx *faissIndex) Search(x []float32, k int64) (
	distances []float32, labels []int64, err error,
) {
	n := len(x) / idx.D()
	distances = make([]float32, int64(n)*k)
	labels = make([]int64, int64(n)*k)
	if c := C.faiss_Index_search(
		idx.idx,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		err = newFaissError(ErrSearchFailed, getLastError(), int(c))
	}

	return
}

func (idx *faissIndex) SearchWithOptions(x []float32, k int64, sel Selector, params json.RawMessage) ([]float32, []int64, error) {
	if sel == nil && params == nil && !idx.HasRaBitQ() {
		return idx.Search(x, k)
	}
	return idx.searchWithOptions(x, k, sel, params)
}

func (idx *faissIndex) Reconstruct(key int64) (recons []float32, err error) {
	rv := make([]float32, idx.D())
	if c := C.faiss_Index_reconstruct(
		idx.idx,
		C.idx_t(key),
		(*C.float)(&rv[0]),
	); c != 0 {
		err = newFaissError(ErrReconstructFailed, getLastError(), int(c))
	}

	return rv, err
}

func (idx *faissIndex) ReconstructBatch(keys []int64, recons []float32) ([]float32, error) {
	var err error
	n := int64(len(keys))
	if c := C.faiss_Index_reconstruct_batch(
		idx.idx,
		C.idx_t(n),
		(*C.idx_t)(&keys[0]),
		(*C.float)(&recons[0]),
	); c != 0 {
		err = newFaissError(ErrReconstructFailed, getLastError(), int(c))
	}

	return recons, err
}

func (idx *faissIndex) MergeFrom(other Index, add_id int64) (err error) {
	// currrently we support the mergeFrom API only for IVF and SQ indexes
	// todo: support on Flat index as well
	if !(idx.IsIVFIndex() && other.IsIVFIndex()) &&
		!(idx.IsSQIndex() && other.IsSQIndex()) {
		return ErrMergeFromNotSupported
	}

	if c := C.faiss_Index_merge_from(
		idx.cPtr(),
		other.cPtr(),
		(C.idx_t)(add_id),
	); c != 0 {
		err = newFaissError(ErrMergeFromFailed, getLastError(), int(c))
	}

	return err
}

func (idx *faissIndex) RangeSearch(x []float32, radius float32) (
	*RangeSearchResult, error,
) {
	n := len(x) / idx.D()
	var rsr *C.FaissRangeSearchResult
	if c := C.faiss_RangeSearchResult_new(&rsr, C.idx_t(n)); c != 0 {
		return nil, newFaissError(ErrSearchFailed, getLastError(), int(c))
	}
	if c := C.faiss_Index_range_search(
		idx.idx,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.float(radius),
		rsr,
	); c != 0 {
		return nil, newFaissError(ErrSearchFailed, getLastError(), int(c))
	}
	return &RangeSearchResult{rsr}, nil
}

func (idx *faissIndex) DistCompute(queryData []float32, ids []int64) ([]float32, error) {
	distances := make([]float32, len(ids))
	if c := C.faiss_Index_dist_compute(idx.idx, (*C.float)(&queryData[0]),
		(*C.idx_t)(&ids[0]), (C.size_t)(len(ids)), (*C.float)(&distances[0])); c != 0 {
		return nil, newFaissError(ErrSearchFailed, getLastError(), int(c))
	}

	return distances, nil
}

func (idx *faissIndex) Reset() error {
	if c := C.faiss_Index_reset(idx.idx); c != 0 {
		return newFaissError(ErrResetIndexFailed, getLastError(), int(c))
	}
	return nil
}

func (idx *faissIndex) RemoveIDs(sel *IDSelector) (int, error) {
	var nRemoved C.size_t
	if c := C.faiss_Index_remove_ids(idx.idx, sel.sel, &nRemoved); c != 0 {
		return 0, newFaissError(ErrRemoveIDsFailed, getLastError(), int(c))
	}
	return int(nRemoved), nil
}

func (idx *faissIndex) Close() {
	C.faiss_Index_free(idx.idx)
}

func (idx *faissIndex) searchWithOptions(x []float32, k int64, sel Selector, params json.RawMessage) ([]float32, []int64, error) {
	// Build a search params object to contain either the selector, the additional params, or both.
	searchParams, err := NewSearchParams(idx, params, sel, nil)
	if err != nil {
		return nil, nil, err
	}
	defer searchParams.Delete()

	n := len(x) / idx.D()
	distances := make([]float32, int64(n)*k)
	labels := make([]int64, int64(n)*k)

	if c := C.faiss_Index_search_with_params(
		idx.idx,
		C.idx_t(n),
		(*C.float)(&x[0]),
		C.idx_t(k),
		searchParams.sp,
		(*C.float)(&distances[0]),
		(*C.idx_t)(&labels[0]),
	); c != 0 {
		return nil, nil, newFaissError(ErrSearchFailed, getLastError(), int(c))
	}
	return distances, labels, nil
}

// -----------------------------------------------------------------------------

// RangeSearchResult is the result of a range search.
type RangeSearchResult struct {
	rsr *C.FaissRangeSearchResult
}

// Nq returns the number of queries.
func (r *RangeSearchResult) Nq() int {
	return int(C.faiss_RangeSearchResult_nq(r.rsr))
}

// Lims returns a slice containing start and end indices for queries in the
// distances and labels slices returned by Labels.
func (r *RangeSearchResult) Lims() []int {
	var lims *C.size_t
	C.faiss_RangeSearchResult_lims(r.rsr, &lims)
	length := r.Nq() + 1
	return (*[1 << 30]int)(unsafe.Pointer(lims))[:length:length]
}

// Labels returns the unsorted IDs and respective distances for each query.
// The result for query i is labels[lims[i]:lims[i+1]].
func (r *RangeSearchResult) Labels() (labels []int64, distances []float32) {
	lims := r.Lims()
	length := lims[len(lims)-1]
	var clabels *C.idx_t
	var cdist *C.float
	C.faiss_RangeSearchResult_labels(r.rsr, &clabels, &cdist)
	labels = (*[1 << 30]int64)(unsafe.Pointer(clabels))[:length:length]
	distances = (*[1 << 30]float32)(unsafe.Pointer(cdist))[:length:length]
	return
}

// Delete frees the memory associated with r.
func (r *RangeSearchResult) Delete() {
	C.faiss_RangeSearchResult_free(r.rsr)
}

// IndexImpl is an abstract structure for an index.
type IndexImpl struct {
	Index
}

// IndexFactory builds a composite index.
// description is a comma-separated list of components.
func IndexFactory(d int, description string, metric int) (*IndexImpl, error) {
	cdesc := C.CString(description)
	defer C.free(unsafe.Pointer(cdesc))
	var idx faissIndex
	c := C.faiss_index_factory(&idx.idx, C.int(d), cdesc, C.FaissMetricType(metric))
	if c != 0 {
		return nil, newFaissError(ErrCreateIndexFailed, getLastError(), int(c))
	}
	return &IndexImpl{&idx}, nil
}

func SetOMPThreads(n uint) {
	C.faiss_set_omp_threads(C.uint(n))
}
