# -*- coding: UTF-8 -*-
import faiss
import torch
from torch import nn
import gc
import logging
import math
import numpy as np
import os
import re
import resource
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from prettytable import PrettyTable
import warnings
import cv2
from os.path import join
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib.pyplot as plt


GIB = 1024 ** 3
MIN_DATABASE_CHUNK_SIZE = 4096
MAX_DATABASE_CHUNK_SIZE = 32768
TARGET_DATABASE_CHUNKS = 100
MEMORY_SAFETY_FACTOR = 1.25
PROGRESS_LOG_INTERVAL_SECONDS = 300
PROGRESS_LOG_INTERVAL_BATCHES = 1000


def _progress(iterable, **kwargs):
    """Keep tqdm interactive while avoiding control characters in redirected logs."""
    kwargs.setdefault("disable", not sys.stderr.isatty())
    return tqdm(iterable, **kwargs)


def _should_log_periodic_progress(batch_number, start_time, last_log_time):
    if sys.stderr.isatty():
        return False
    if batch_number == 1:
        return True
    if batch_number % PROGRESS_LOG_INTERVAL_BATCHES == 0:
        return True
    return time.perf_counter() - last_log_time >= PROGRESS_LOG_INTERVAL_SECONDS


def match_batch_tensor(fm1, fm2, trainflag, grid_size, T2=0.7):
    '''
    fm1: (l,D) 529,768
    fm2: (N,l,D) 100,529,768
    mask1: (l)
    mask2: (N,l)
    '''
    M = torch.matmul(fm2, fm1.T)

    max1 = torch.argmax(M, dim=1)
    max2 = torch.argmax(M, dim=2)
    batch_indexes = torch.arange(M.shape[0], device=M.device).reshape((-1, 1))
    m = max2[batch_indexes, max1]
    valid = torch.arange(M.shape[-1], device=M.device).repeat((M.shape[0], 1)) == m
    scores = torch.zeros(fm2.shape[0], device=M.device)

    for i in range(fm2.shape[0]):
        idx1 = torch.nonzero(valid[i, :]).squeeze()
        idx2 = max1[i, :][idx1]
        assert idx1.shape == idx2.shape

        if len(idx1.shape) > 0:
            # Calculate cosine similarity and apply threshold
            cos_similarity = torch.sum(fm1[idx1] * fm2[i][idx2], dim=1)
            valid_pairs = cos_similarity > T2
            idx1 = idx1[valid_pairs]
            idx2 = idx2[valid_pairs]

        if trainflag:
            if len(idx1.shape) > 0:
                similarity = torch.mean(torch.sum(fm1[idx1] * fm2[i][idx2], dim=1), dim=0)
            else:
                print("No mutual nearest neighbors!")
                similarity = torch.mean(torch.sum(fm1 * fm2[i], dim=1), dim=0)
            return similarity

        else:
            if len(idx1.shape) < 1:
                scores[i] = 0
            else:
                scores[i] = len(idx1)
    return scores

def local_sim(features_1, features_2, trainflag=False):
    B, Num, C = features_2.shape
    if trainflag:
        queries = features_1
        preds = features_2
        similarity = torch.zeros(B, device=features_2.device)
        for i in range(B):
            query,pred = queries[i],preds[i].unsqueeze(0)
            similarity[i] = match_batch_tensor(query, pred, trainflag, grid_size=(61,61))
        return similarity
    else:
        query = features_1
        preds = features_2
        scores = match_batch_tensor(query, preds,trainflag, grid_size=(61,61))
        return scores


def rerank(predictions, queries_local_features, database_local_features):
    pred2 = []
    print("reranking...")
    for query_index, pred in enumerate(_progress(predictions)):
        query_local_features = torch.tensor(queries_local_features[query_index]).cuda()
        positives_local_features = torch.tensor(database_local_features[pred]).cuda()
        rerank_index = local_sim(query_local_features, positives_local_features, trainflag=False)
        rerank_index_sorted = rerank_index.cpu().numpy().argsort()[::-1]
        pred2.append(predictions[query_index][rerank_index_sorted])
    return np.array(pred2)


def compute_recalls(args, eval_ds, predictions):
    """Compute recalls for the full query set or for each configured query group."""
    positives_per_query = eval_ds.get_positives()

    def compute_for_slice(start_index, end_index):
        sliced_predictions = predictions[start_index:end_index]
        sliced_positives = positives_per_query[start_index:end_index]
        recalls = np.zeros(len(args.recall_values))
        for pred, positives in zip(sliced_predictions, sliced_positives):
            for i, n in enumerate(args.recall_values):
                if np.any(np.in1d(pred[:n], positives)):
                    recalls[i:] += 1
                    break
        recalls = recalls / len(sliced_predictions) * 100
        recalls_str = ", ".join(
            f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)
        )
        return recalls, recalls_str

    query_group_slices = getattr(eval_ds, "query_group_slices", None)
    if not query_group_slices:
        return compute_for_slice(0, eval_ds.queries_num)

    grouped_recalls = OrderedDict()
    grouped_recalls_str = OrderedDict()
    for query_set_name, (start_index, end_index) in query_group_slices.items():
        recalls, recalls_str = compute_for_slice(start_index, end_index)
        grouped_recalls[query_set_name] = recalls
        grouped_recalls_str[query_set_name] = recalls_str
    return grouped_recalls, grouped_recalls_str


def print_recalls_table(args, eval_ds, recalls, title):
    recall_groups = recalls.items() if isinstance(recalls, dict) else [(eval_ds, recalls)]
    for dataset_label, query_set_recalls in recall_groups:
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in args.recall_values]
        table.add_row(['Recall@K'] + [f'{v:.2f}' for v in query_set_recalls])
        print(table.get_string(title=f"{title} on {dataset_label}"))


def _format_bytes(num_bytes):
    return f"{num_bytes / GIB:.2f} GiB"


def _get_current_rss_bytes():
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as statm_file:
            resident_pages = int(statm_file.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _get_available_memory_bytes():
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as meminfo_file:
            for line in meminfo_file:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def _get_database_chunk_size(database_num, infer_batch_size):
    if database_num <= MIN_DATABASE_CHUNK_SIZE:
        return database_num
    target_size = math.ceil(database_num / TARGET_DATABASE_CHUNKS)
    target_size = max(MIN_DATABASE_CHUNK_SIZE, min(MAX_DATABASE_CHUNK_SIZE, target_size))
    aligned_size = math.ceil(target_size / infer_batch_size) * infer_batch_size
    return min(aligned_size, database_num)


class _DiskCacheGuard:
    def __init__(self, args, dataset_name):
        self.root = os.path.abspath(args.efficient_ram_cache_dir)
        self.max_cache_bytes = int(args.efficient_ram_max_cache_gib * GIB)
        self.min_free_bytes = int(args.efficient_ram_min_free_gib * GIB)
        self.reserved_bytes = 0
        self.work_dir = None

        if self.max_cache_bytes <= 0:
            raise ValueError("--efficient_ram_max_cache_gib must be greater than zero")
        if self.min_free_bytes < 0:
            raise ValueError("--efficient_ram_min_free_gib cannot be negative")
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Low-memory cache directory does not exist: {self.root}")
        if not os.access(self.root, os.R_OK | os.W_OK | os.X_OK):
            raise PermissionError(f"Low-memory cache directory is not writable: {self.root}")

        self._check_free_space("before creating the cache directory")
        safe_dataset_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset_name))
        self.work_dir = tempfile.mkdtemp(prefix=f"fol_{safe_dataset_name}_", dir=self.root)
        logging.info(f"Low-memory cache directory: {self.work_dir}")

    def _free_bytes(self):
        return shutil.disk_usage(self.root).free

    def _check_free_space(self, context):
        free_bytes = self._free_bytes()
        if free_bytes <= self.min_free_bytes:
            raise RuntimeError(
                f"Disk safety limit reached {context}: available={_format_bytes(free_bytes)}, "
                f"required to remain above {_format_bytes(self.min_free_bytes)}"
            )
        return free_bytes

    def ensure_total_capacity(self, total_bytes, context):
        if total_bytes >= self.max_cache_bytes:
            raise RuntimeError(
                f"Cache safety limit would be reached {context}: required={_format_bytes(total_bytes)}, "
                f"limit={_format_bytes(self.max_cache_bytes)}"
            )
        additional_bytes = max(0, total_bytes - self.reserved_bytes)
        free_bytes = self._check_free_space(context)
        if free_bytes - additional_bytes <= self.min_free_bytes:
            raise RuntimeError(
                f"Disk reserve would be violated {context}: available={_format_bytes(free_bytes)}, "
                f"additional cache={_format_bytes(additional_bytes)}, "
                f"minimum free={_format_bytes(self.min_free_bytes)}"
            )

    def reserve_memmap(self, filename, shape, dtype=np.float32):
        dtype = np.dtype(dtype)
        size_bytes = math.prod(shape) * dtype.itemsize
        total_bytes = self.reserved_bytes + size_bytes
        self.ensure_total_capacity(total_bytes, f"while reserving {filename}")

        path = os.path.join(self.work_dir, filename)
        file_descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            try:
                os.posix_fallocate(file_descriptor, 0, size_bytes)
            except AttributeError as error:
                raise RuntimeError(
                    "This platform does not support physical cache preallocation with posix_fallocate"
                ) from error
        except BaseException:
            os.close(file_descriptor)
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        else:
            os.close(file_descriptor)

        self.reserved_bytes = total_bytes
        self.check_runtime(f"after reserving {filename}")
        logging.info(
            f"Reserved {filename}: {_format_bytes(size_bytes)}; "
            f"total cache={_format_bytes(self.reserved_bytes)}"
        )
        return np.memmap(path, dtype=dtype, mode="r+", shape=shape)

    def check_runtime(self, context):
        if self.reserved_bytes >= self.max_cache_bytes:
            raise RuntimeError(
                f"Cache safety limit reached {context}: cache={_format_bytes(self.reserved_bytes)}, "
                f"limit={_format_bytes(self.max_cache_bytes)}"
            )
        self._check_free_space(context)

    def cleanup(self):
        if self.work_dir is None:
            return
        try:
            shutil.rmtree(self.work_dir)
            logging.info(f"Removed low-memory cache directory: {self.work_dir}")
        except FileNotFoundError:
            pass
        except OSError as error:
            logging.warning(f"Could not fully remove low-memory cache {self.work_dir}: {error}")
        finally:
            self.work_dir = None


class _OriginalBatchSampler:
    """Yield selected batches with the same boundaries as the full evaluation dataset."""
    def __init__(self, batch_starts, batch_size, dataset_size):
        self.batch_starts = batch_starts
        self.batch_size = batch_size
        self.dataset_size = dataset_size

    def __iter__(self):
        for start_index in self.batch_starts:
            start_index = int(start_index)
            yield range(start_index, min(start_index + self.batch_size, self.dataset_size))

    def __len__(self):
        return len(self.batch_starts)


def _make_original_batch_dataloader(args, eval_ds, batch_starts):
    batch_sampler = _OriginalBatchSampler(
        batch_starts, args.infer_batch_size, len(eval_ds)
    )
    return DataLoader(
        dataset=eval_ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )


def _model_features_and_local(args, model, inputs, pca=None):
    outputs = model(inputs.to(args.device), test=True)
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise ValueError("The model must return global and local features during evaluation")
    features, local_features = outputs[0], outputs[1]
    features = features.cpu().numpy()
    if pca is not None:
        features = pca.transform(features)
    features = np.asarray(features, dtype=np.float32)
    local_features = np.asarray(local_features.cpu().numpy(), dtype=np.float32)
    return features, local_features


def _report_efficient_ram_requirements(args, eval_ds, guard, local_shape,
                                       chunk_size, search_k):
    local_length, local_dim = local_shape
    float_size = np.dtype(np.float32).itemsize
    index_size = np.dtype(np.int64).itemsize
    query_global_bytes = eval_ds.queries_num * args.features_dim * float_size
    local_image_bytes = local_length * local_dim * float_size
    query_local_bytes = eval_ds.queries_num * local_image_bytes
    worst_candidate_num = min(
        eval_ds.database_num, eval_ds.queries_num * search_k
    )
    worst_candidate_forward_num = min(
        eval_ds.database_num, worst_candidate_num * args.infer_batch_size
    )
    candidate_local_bytes = worst_candidate_num * local_image_bytes
    worst_cache_bytes = query_local_bytes + candidate_local_bytes

    database_chunk_bytes = chunk_size * args.features_dim * float_size
    topk_bytes = eval_ds.queries_num * search_k * (float_size + index_size)
    rerank_cpu_bytes = 2 * (search_k + 1) * local_image_bytes
    minimum_extra_bytes = (
        query_global_bytes + 2 * database_chunk_bytes + 7 * topk_bytes + rerank_cpu_bytes
    )
    recommended_bytes = math.ceil(minimum_extra_bytes * MEMORY_SAFETY_FACTOR)
    current_rss_bytes = _get_current_rss_bytes()
    available_memory_bytes = _get_available_memory_bytes()
    disk_free_bytes = shutil.disk_usage(guard.root).free
    chunks_num = math.ceil(eval_ds.database_num / chunk_size)

    rerank_matrix_bytes = search_k * local_length * local_length * float_size
    rerank_gpu_bytes = rerank_matrix_bytes + (search_k + 1) * local_image_bytes

    logging.info("Exact low-memory evaluation enabled")
    logging.info(
        f"Evaluation size: database={eval_ds.database_num}, queries={eval_ds.queries_num}, "
        f"global_dim={args.features_dim}, local_shape={local_shape}, top_k={search_k}"
    )
    logging.info(
        f"Database chunk: {chunk_size} images, approximately {chunks_num} chunks"
    )
    logging.info(
        f"Memory: process RSS={_format_bytes(current_rss_bytes)}, "
        f"system available={_format_bytes(available_memory_bytes)}"
    )
    logging.info(
        f"Estimated extra CPU memory: minimum={_format_bytes(minimum_extra_bytes)}, "
        f"recommended={_format_bytes(recommended_bytes)}"
    )
    logging.info(
        f"Estimated reranking GPU workspace: approximately {_format_bytes(rerank_gpu_bytes)} "
        "excluding the model and CUDA allocator overhead"
    )
    logging.info(
        f"Disk: available={_format_bytes(disk_free_bytes)}, "
        f"query cache={_format_bytes(query_local_bytes)}, "
        f"worst candidate cache={_format_bytes(candidate_local_bytes)}, "
        f"worst total cache={_format_bytes(worst_cache_bytes)}"
    )
    logging.info(
        f"Disk limits: cache must stay below {_format_bytes(guard.max_cache_bytes)} and "
        f"free space must stay above {_format_bytes(guard.min_free_bytes)}"
    )
    logging.info(
        f"Worst-case cached candidates: {worst_candidate_num}; "
        f"full-batch candidate forwards: {worst_candidate_forward_num} images "
        f"({100 * worst_candidate_forward_num / eval_ds.database_num:.2f}% of the database)"
    )
    logging.info("Estimates exclude model, DataLoader workers, CUDA context, and OS page cache")

    if available_memory_bytes < minimum_extra_bytes:
        raise MemoryError(
            f"Available memory {_format_bytes(available_memory_bytes)} is below the "
            f"estimated minimum {_format_bytes(minimum_extra_bytes)}"
        )
    if available_memory_bytes < recommended_bytes:
        logging.warning(
            f"Available memory is below the recommended {_format_bytes(recommended_bytes)}"
        )

    if args.device == "cuda" and torch.cuda.is_available():
        cuda_free_bytes, cuda_total_bytes = torch.cuda.mem_get_info()
        logging.info(
            f"CUDA memory: free={_format_bytes(cuda_free_bytes)}, "
            f"total={_format_bytes(cuda_total_bytes)}"
        )
        if cuda_free_bytes < rerank_gpu_bytes:
            logging.warning(
                "Current free CUDA memory is below the estimated reranking workspace; "
                "reranking may run out of GPU memory"
            )

    guard.ensure_total_capacity(worst_cache_bytes, "for the worst-case cache estimate")
    return query_local_bytes, candidate_local_bytes


def _extract_query_features_to_cache(args, eval_ds, model, pca, guard, search_k,
                                     chunk_size):
    queries_features = np.empty(
        (eval_ds.queries_num, args.features_dim), dtype=np.float32
    )
    query_local_cache = None
    local_shape = None
    written_queries = np.zeros(eval_ds.queries_num, dtype=bool)

    eval_ds.test_method = "hard_resize"
    first_batch_start = (eval_ds.database_num // args.infer_batch_size) * args.infer_batch_size
    batch_starts = range(first_batch_start, len(eval_ds), args.infer_batch_size)
    dataloader = _make_original_batch_dataloader(args, eval_ds, batch_starts)

    logging.info("Extracting query global and local features with original batch boundaries")
    progress_start_time = time.perf_counter()
    last_progress_log_time = progress_start_time
    for batch_number, (inputs, indices) in enumerate(
            _progress(dataloader, ncols=100, desc="Queries"), start=1):
        guard.check_runtime("before a query inference batch")
        features, local_features = _model_features_and_local(args, model, inputs, pca)
        indices = indices.numpy()
        query_positions = np.flatnonzero(indices >= eval_ds.database_num)
        if len(query_positions) == 0:
            continue

        if local_shape is None:
            local_shape = tuple(local_features.shape[1:])
            _report_efficient_ram_requirements(
                args, eval_ds, guard, local_shape, chunk_size, search_k
            )
            query_local_cache = guard.reserve_memmap(
                "query_local.float32.dat",
                (eval_ds.queries_num,) + local_shape,
                np.float32,
            )
        elif tuple(local_features.shape[1:]) != local_shape:
            raise RuntimeError(
                f"Local feature shape changed from {local_shape} to "
                f"{tuple(local_features.shape[1:])}"
            )

        query_indices = indices[query_positions] - eval_ds.database_num
        queries_features[query_indices] = features[query_positions]
        query_local_cache[query_indices] = local_features[query_positions]
        written_queries[query_indices] = True
        guard.check_runtime("after a query cache write")
        if _should_log_periodic_progress(
                batch_number, progress_start_time, last_progress_log_time):
            cached_queries = int(np.count_nonzero(written_queries))
            logging.info(
                f"Query extraction progress: cached {cached_queries}/"
                f"{eval_ds.queries_num} queries after {batch_number} batches; "
                f"elapsed={time.perf_counter() - progress_start_time:.0f}s"
            )
            last_progress_log_time = time.perf_counter()

    if query_local_cache is None or not np.all(written_queries):
        missing_num = int(np.count_nonzero(~written_queries))
        raise RuntimeError(f"Failed to cache {missing_num} query local features")
    query_local_cache.flush()
    return queries_features, query_local_cache, local_shape


def _merge_exact_topk(best_distances, best_predictions, chunk_distances,
                      chunk_predictions, search_k):
    candidate_distances = np.concatenate((best_distances, chunk_distances), axis=1)
    candidate_predictions = np.concatenate((best_predictions, chunk_predictions), axis=1)
    # Match IndexFlatL2 ordering for exact ties by preferring the lower database index.
    order = np.lexsort((candidate_predictions, candidate_distances), axis=1)[:, :search_k]
    best_distances = np.take_along_axis(candidate_distances, order, axis=1)
    best_predictions = np.take_along_axis(candidate_predictions, order, axis=1)
    return best_distances, best_predictions


def _search_database_in_chunks(args, eval_ds, model, pca, guard,
                               queries_features, chunk_size, search_k):
    invalid_index = np.iinfo(np.int64).max
    best_distances = np.full(
        (eval_ds.queries_num, search_k), np.inf, dtype=np.float32
    )
    best_predictions = np.full(
        (eval_ds.queries_num, search_k), invalid_index, dtype=np.int64
    )
    database_chunk = np.empty((chunk_size, args.features_dim), dtype=np.float32)
    chunk_count = 0
    chunk_start_index = 0
    next_database_index = 0

    def search_current_chunk(count):
        nonlocal best_distances, best_predictions, chunk_start_index
        chunk_features = np.ascontiguousarray(database_chunk[:count])
        chunk_index = faiss.IndexFlatL2(args.features_dim)
        chunk_index.add(chunk_features)
        chunk_k = min(search_k, count)
        chunk_distances, chunk_predictions = chunk_index.search(
            queries_features, chunk_k
        )
        chunk_predictions += chunk_start_index
        del chunk_index

        if chunk_k < search_k:
            pad_width = search_k - chunk_k
            chunk_distances = np.pad(
                chunk_distances, ((0, 0), (0, pad_width)), constant_values=np.inf
            )
            chunk_predictions = np.pad(
                chunk_predictions,
                ((0, 0), (0, pad_width)),
                constant_values=invalid_index,
            )
        best_distances, best_predictions = _merge_exact_topk(
            best_distances, best_predictions, chunk_distances,
            chunk_predictions, search_k
        )
        chunk_start_index += count

    eval_ds.test_method = "hard_resize"
    database_batch_end = min(
        len(eval_ds),
        math.ceil(eval_ds.database_num / args.infer_batch_size) * args.infer_batch_size,
    )
    batch_starts = range(0, database_batch_end, args.infer_batch_size)
    dataloader = _make_original_batch_dataloader(args, eval_ds, batch_starts)

    logging.info("Extracting and searching database global features in exact chunks")
    progress_start_time = time.perf_counter()
    last_progress_log_time = progress_start_time
    for batch_number, (inputs, indices) in enumerate(
            _progress(dataloader, ncols=100, desc="Database retrieval"), start=1):
        guard.check_runtime("before a database retrieval batch")
        outputs = model(inputs.to(args.device), test=True)
        features = outputs[0].cpu().numpy()
        if pca is not None:
            features = pca.transform(features)
        features = np.asarray(features, dtype=np.float32)

        indices = indices.numpy()
        database_positions = np.flatnonzero(indices < eval_ds.database_num)
        database_indices = indices[database_positions]
        expected_indices = np.arange(
            next_database_index, next_database_index + len(database_indices)
        )
        if not np.array_equal(database_indices, expected_indices):
            raise RuntimeError("Database DataLoader returned non-contiguous indexes")
        next_database_index += len(database_indices)
        batch_features = features[database_positions]

        batch_offset = 0
        while batch_offset < len(batch_features):
            copy_count = min(chunk_size - chunk_count, len(batch_features) - batch_offset)
            database_chunk[chunk_count:chunk_count + copy_count] = \
                batch_features[batch_offset:batch_offset + copy_count]
            chunk_count += copy_count
            batch_offset += copy_count
            if chunk_count == chunk_size:
                search_current_chunk(chunk_count)
                chunk_count = 0
        guard.check_runtime("after a database retrieval batch")
        if _should_log_periodic_progress(
                batch_number, progress_start_time, last_progress_log_time):
            completed_chunks = chunk_start_index // chunk_size
            logging.info(
                f"Database retrieval progress: processed {next_database_index}/"
                f"{eval_ds.database_num} database images; searched "
                f"{completed_chunks}/{math.ceil(eval_ds.database_num / chunk_size)} "
                f"chunks; elapsed={time.perf_counter() - progress_start_time:.0f}s"
            )
            last_progress_log_time = time.perf_counter()

    if next_database_index != eval_ds.database_num:
        raise RuntimeError(
            f"Expected {eval_ds.database_num} database features, got {next_database_index}"
        )
    if chunk_count:
        search_current_chunk(chunk_count)
    return best_predictions


def _extract_candidate_local_features(args, eval_ds, model, guard, predictions,
                                      local_shape):
    unique_candidate_ids = np.unique(predictions.reshape(-1))
    unique_candidate_ids = unique_candidate_ids[
        (unique_candidate_ids >= 0) & (unique_candidate_ids < eval_ds.database_num)
    ]
    candidate_cache_bytes = (
        len(unique_candidate_ids) * math.prod(local_shape) * np.dtype(np.float32).itemsize
    )
    total_cache_bytes = guard.reserved_bytes + candidate_cache_bytes
    guard.ensure_total_capacity(total_cache_bytes, "for the actual candidate cache")
    logging.info(
        f"Unique reranking candidates: {len(unique_candidate_ids)}; "
        f"actual candidate cache={_format_bytes(candidate_cache_bytes)}; "
        f"actual total cache={_format_bytes(total_cache_bytes)}"
    )

    candidate_local_cache = guard.reserve_memmap(
        "candidate_local.float32.dat",
        (len(unique_candidate_ids),) + local_shape,
        np.float32,
    )
    written_candidates = np.zeros(len(unique_candidate_ids), dtype=bool)

    batch_starts = np.unique(
        (unique_candidate_ids // args.infer_batch_size) * args.infer_batch_size
    )
    forwarded_images_num = sum(
        min(int(start_index) + args.infer_batch_size, len(eval_ds)) - int(start_index)
        for start_index in batch_starts
    )
    logging.info(
        f"Candidate re-extraction: {len(batch_starts)} original batches, "
        f"{forwarded_images_num} total images"
    )
    eval_ds.test_method = "hard_resize"
    dataloader = _make_original_batch_dataloader(args, eval_ds, batch_starts)

    logging.info("Re-extracting local features only for original batches containing candidates")
    progress_start_time = time.perf_counter()
    last_progress_log_time = progress_start_time
    for batch_number, (inputs, indices) in enumerate(
            _progress(dataloader, ncols=100, desc="Candidate local features"),
            start=1):
        guard.check_runtime("before a candidate inference batch")
        outputs = model(inputs.to(args.device), test=True)
        local_features = np.asarray(outputs[1].cpu().numpy(), dtype=np.float32)
        if tuple(local_features.shape[1:]) != local_shape:
            raise RuntimeError(
                f"Local feature shape changed from {local_shape} to "
                f"{tuple(local_features.shape[1:])}"
            )

        indices = indices.numpy()
        database_positions = np.flatnonzero(indices < eval_ds.database_num)
        database_indices = indices[database_positions]
        candidate_rows = np.searchsorted(unique_candidate_ids, database_indices)
        in_range = candidate_rows < len(unique_candidate_ids)
        matched = np.zeros(len(candidate_rows), dtype=bool)
        matched[in_range] = (
            unique_candidate_ids[candidate_rows[in_range]] == database_indices[in_range]
        )
        if np.any(matched):
            rows = candidate_rows[matched]
            model_positions = database_positions[matched]
            candidate_local_cache[rows] = local_features[model_positions]
            written_candidates[rows] = True
        guard.check_runtime("after a candidate cache write")
        if _should_log_periodic_progress(
                batch_number, progress_start_time, last_progress_log_time):
            cached_candidates = int(np.count_nonzero(written_candidates))
            logging.info(
                f"Candidate local progress: cached {cached_candidates}/"
                f"{len(unique_candidate_ids)} candidates after {batch_number}/"
                f"{len(batch_starts)} original batches; "
                f"elapsed={time.perf_counter() - progress_start_time:.0f}s"
            )
            last_progress_log_time = time.perf_counter()

    if not np.all(written_candidates):
        missing_num = int(np.count_nonzero(~written_candidates))
        raise RuntimeError(f"Failed to cache {missing_num} candidate local features")
    candidate_local_cache.flush()
    return unique_candidate_ids, candidate_local_cache


def _rerank_from_disk_cache(args, predictions, query_local_cache,
                            unique_candidate_ids, candidate_local_cache, guard):
    reranked_predictions = []
    candidate_rows = np.searchsorted(unique_candidate_ids, predictions)
    if not np.array_equal(unique_candidate_ids[candidate_rows], predictions):
        raise RuntimeError("Candidate cache does not cover all retrieval predictions")

    query_shape = tuple(query_local_cache.shape[1:])
    candidate_shape = (predictions.shape[1],) + query_shape
    is_cuda = torch.device(args.device).type == "cuda"
    use_pinned_memory = is_cuda
    try:
        query_cpu_staging = torch.empty(
            query_shape, dtype=torch.float32, pin_memory=use_pinned_memory
        )
        candidate_cpu_staging = torch.empty(
            candidate_shape, dtype=torch.float32, pin_memory=use_pinned_memory
        )
    except RuntimeError as error:
        if not use_pinned_memory:
            raise
        logging.warning(
            f"Could not allocate pinned reranking staging buffers ({error}); "
            "falling back to regular CPU memory"
        )
        use_pinned_memory = False
        query_cpu_staging = torch.empty(query_shape, dtype=torch.float32)
        candidate_cpu_staging = torch.empty(candidate_shape, dtype=torch.float32)

    query_cpu_numpy = query_cpu_staging.numpy()
    candidate_cpu_numpy = candidate_cpu_staging.numpy()
    query_device_staging = torch.empty(
        query_shape, dtype=torch.float32, device=args.device
    )
    candidate_device_staging = torch.empty(
        candidate_shape, dtype=torch.float32, device=args.device
    )
    staging_bytes = query_cpu_staging.numel() * query_cpu_staging.element_size()
    staging_bytes += (
        candidate_cpu_staging.numel() * candidate_cpu_staging.element_size()
    )
    logging.info(
        f"Reranking staging buffers: {_format_bytes(staging_bytes)} CPU "
        f"({'pinned' if use_pinned_memory else 'regular'}) and "
        f"{_format_bytes(staging_bytes)} on {args.device}"
    )

    print("reranking...")
    for query_index, prediction in enumerate(_progress(predictions)):
        guard.check_runtime("before reranking a query")

        gather_start = time.perf_counter()
        np.copyto(query_cpu_numpy, query_local_cache[query_index])
        for output_row, cache_row in enumerate(candidate_rows[query_index]):
            np.copyto(
                candidate_cpu_numpy[output_row],
                candidate_local_cache[cache_row],
            )
        gather_seconds = time.perf_counter() - gather_start
        if query_index == 0:
            logging.info(
                f"First rerank query cache gather completed in "
                f"{gather_seconds:.3f}s"
            )

        transfer_start = time.perf_counter()
        query_device_staging.copy_(
            query_cpu_staging, non_blocking=use_pinned_memory
        )
        candidate_device_staging.copy_(
            candidate_cpu_staging, non_blocking=use_pinned_memory
        )
        if is_cuda:
            torch.cuda.synchronize(query_device_staging.device)
        transfer_seconds = time.perf_counter() - transfer_start
        if query_index == 0:
            logging.info(
                f"First rerank query CPU-to-{args.device} transfer completed in "
                f"{transfer_seconds:.3f}s"
            )

        similarity_start = time.perf_counter()
        rerank_scores = local_sim(
            query_device_staging, candidate_device_staging, trainflag=False
        )
        rerank_scores_numpy = rerank_scores.cpu().numpy()
        similarity_seconds = time.perf_counter() - similarity_start
        if query_index == 0:
            logging.info(
                f"First rerank query local_sim completed in "
                f"{similarity_seconds:.3f}s"
            )
        rerank_order = rerank_scores_numpy.argsort()[::-1]
        reranked_predictions.append(prediction[rerank_order])
        guard.check_runtime("after reranking a query")

        completed_queries = query_index + 1
        if (completed_queries == 1 or completed_queries % 50 == 0 or
                completed_queries == len(predictions)):
            logging.info(
                f"Rerank timing query {completed_queries}/{len(predictions)}: "
                f"cache gather={gather_seconds:.3f}s, "
                f"CPU-to-{args.device}={transfer_seconds:.3f}s, "
                f"local_sim={similarity_seconds:.3f}s"
            )
    return np.asarray(reranked_predictions)


def _close_memmap(memmap):
    if memmap is None:
        return
    try:
        memmap.flush()
    except (OSError, ValueError) as error:
        logging.warning(f"Could not flush a low-memory cache before cleanup: {error}")
    try:
        memmap._mmap.close()
    except (AttributeError, OSError, ValueError):
        pass


def test_efficient_ram_usage(args, eval_ds, model, test_method="hard_resize", pca=None):
    """Run exact retrieval and reranking without retaining all database features in RAM."""
    if eval_ds.database_num == 0 or eval_ds.queries_num == 0:
        raise ValueError("The evaluation dataset must contain database and query images")
    if args.infer_batch_size <= 0:
        raise ValueError("--infer_batch_size must be greater than zero")

    model = model.eval().to(args.device)
    search_k = min(max(args.recall_values), eval_ds.database_num)
    chunk_size = _get_database_chunk_size(
        eval_ds.database_num, args.infer_batch_size
    )
    guard = None
    query_local_cache = None
    candidate_local_cache = None

    try:
        guard = _DiskCacheGuard(args, eval_ds.dataset_name)
        with torch.no_grad():
            queries_features, query_local_cache, local_shape = \
                _extract_query_features_to_cache(
                    args, eval_ds, model, pca, guard, search_k, chunk_size
                )
            predictions = _search_database_in_chunks(
                args, eval_ds, model, pca, guard, queries_features,
                chunk_size, search_k
            )
            del queries_features

            recalls, recalls_str = compute_recalls(args, eval_ds, predictions)
            print()
            print_recalls_table(args, eval_ds, recalls, "Performances")

            unique_candidate_ids, candidate_local_cache = \
                _extract_candidate_local_features(
                    args, eval_ds, model, guard, predictions, local_shape
                )
            if args.device == "cuda":
                torch.cuda.empty_cache()
            predictions_rerank = _rerank_from_disk_cache(
                args, predictions, query_local_cache,
                unique_candidate_ids, candidate_local_cache, guard
            )

            recalls_rerank, recalls_str_rerank = compute_recalls(
                args, eval_ds, predictions_rerank
            )
            print()
            print_recalls_table(
                args, eval_ds, recalls_rerank, "Reranking Performances"
            )
        return recalls, recalls_str, recalls_rerank, recalls_str_rerank
    finally:
        _close_memmap(query_local_cache)
        _close_memmap(candidate_local_cache)
        query_local_cache = None
        candidate_local_cache = None
        gc.collect()
        if guard is not None:
            guard.cleanup()


def test(args, eval_ds, model, test_method="hard_resize", pca=None):
    assert test_method in ["hard_resize", "single_query", "central_crop", "five_crops",
                            "nearest_crop", "maj_voting"], f"test_method can't be {test_method}"

    if args.efficient_ram_testing:
        return test_efficient_ram_usage(args, eval_ds, model, test_method, pca)

    model = model.eval().to(args.device)

    with torch.no_grad():
        logging.debug("Extracting database and queries features for evaluation/testing")
        eval_ds.test_method = "hard_resize"
        dataloader = DataLoader(dataset=eval_ds, num_workers=args.num_workers,
                                batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))

        all_features = torch.empty((len(eval_ds), args.features_dim), dtype=torch.float32, device='cpu')

        for batch_idx, (inputs, indices) in enumerate(
                _progress(dataloader, ncols=100)):
            inputs = inputs.to(args.device)
            outputs = model(inputs, test=True)

            if len(outputs) == 2:
                features, local_f = outputs
            elif len(outputs) != 2:
                features, local_f = outputs[0], outputs[1]
            else:
                raise ValueError("Unexpected number of outputs from model")
            if batch_idx == 0:
                local_f_dim1, local_f_dim2 = local_f.shape[-2], local_f.shape[-1]
                local_f_all = torch.empty((len(eval_ds), local_f_dim1, local_f_dim2), dtype=torch.float32, device='cpu')
            if pca is not None:
                features = torch.from_numpy(pca.transform(features.cpu().numpy())).to(args.device)

            start_idx = batch_idx * args.infer_batch_size
            end_idx = start_idx + len(indices)
            all_features[start_idx:end_idx, :] = features.cpu()
            local_f_all[start_idx:end_idx, :, :] = local_f.cpu()


    queries_features = all_features[eval_ds.database_num:].cpu().numpy()
    database_features = all_features[:eval_ds.database_num].cpu().numpy()
    q_local_list = local_f_all[eval_ds.database_num:].to(torch.float32)
    r_local_list = local_f_all[:eval_ds.database_num].to(torch.float32)

    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features

    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))


    recalls, recalls_str = compute_recalls(args, eval_ds, predictions)

    print()  # print a new line
    print_recalls_table(args, eval_ds, recalls, "Performances")

    # rerank
    predictions2 = rerank(predictions, q_local_list, r_local_list)

    recalls_rerank, recalls_str_rerank = compute_recalls(args, eval_ds, predictions2)

    print()  # print a new line
    print_recalls_table(args, eval_ds, recalls_rerank, "Reranking Performances")

    return recalls, recalls_str, recalls_rerank, recalls_str_rerank
