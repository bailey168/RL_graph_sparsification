# Plan: Top-K FC graph sparsification baseline with a fixed GCN (FC21 / FC55)

## Context
The ECE1508 project will train an RL agent (discrete SAC) that decides, edge by edge, which FC edges each subject keeps, with the goal of better cognitive prediction. Before that, we need the **non-RL baseline the proposal names: edge-weight thresholding**. For each subject we keep the top K% of edges by |FC|, train a fixed and configurable GCN, and measure how prediction changes with sparsity. We run this for all 4 cognitive tests (GF, DSST, PAL, TMT) on both FC representations.

The code is new, lives at the repo root, and takes ideas from `others/` without changing it. The most relevant references are:
- `others/gnn_training_pipeline/GNN_all_scaling_law_optuna_nocv.ipynb`: newest pipeline (CSV → 70/15/15 split → train-only scalers → in-memory PyG graphs → AdamW + warmup/cosine + early stopping)
- `others/process_graph_pipeline/csv_to_graph.py`: `parse_region_map`, per-subject top-k
- `others/gnn/architectures_normal.py`: model skeleton (node-ID embedding → conv stack → attentional pooling → demographics projection → MLP head, Kaiming init)
- `others/gnn_training_pipeline/util/{reproducibility,early_stopping}.py`, `metrics.py` (`calc_r2_corr`)

Step 0 of implementation: copy this plan to `PLAN.md` at the repo root (the user asked for the working plan to live there) and keep it updated as decisions change.

## Decisions (agreed)
| Topic | Decision |
|---|---|
| Naming | **FC21** = UKB FC25 (21 nodes, 210 edges, CSV columns `FC25 (ICa-ICb)`); **FC55** = UKB FC100 (55 nodes, 1485 edges, columns `FC100 (...)`). New names are used everywhere in our code and outputs. The CSV column strings are left unchanged. |
| Data | `/scratch/bng/cartbind/data/UKB_new_data/combined_data_no_outliers/combined_data_{TEST}_no_outliers.csv` (edges, demographics and target in one file). Optional EID list `.../MIND_models/scaling_law_splits/{TEST}/{TEST}_all_eids.txt`. All paths set in YAML. |
| Targets | GF `p20016_i2`, PAL `p20197_i2`, DSST `p23324_i2`, TMT `p6350_i2` |
| Split | 70/15/15 train/val/test, `train_test_split(random_state=42)` as in the notebook, fixed across sparsity levels and FC types. Split EIDs are saved for reuse by the RL stage. Missing values raise an error (notebook behaviour). |
| Sparsification | Two masks are compared (`sweep.masks`). **personal**: each subject keeps `k = max(1, floor(p·E))` edges with the largest **\|raw FC\|** (`np.argpartition`). **shared** (added 2026-10-31): one set of k edges from the train split, applied to every subject. Edges are ranked by **frequency**: how often an edge is in a train subject's personal top-K, with ties broken by mean \|FC\|. One rule per mask, no ranking options (simplified 2026-11-01). |
| Edge value | **Group z-score** of each edge (`edge_value: standardized`, changed 2026-10-31). A `StandardScaler` per edge is fit on the **pre-sparsified** train edges, then applied to all splits. Selection always ranks by raw \|FC\|. Setting `edge_value: raw` still works but is no longer a baseline we run. |
| Self-loops | Kept, with `self_loop_weight` (default 1.0) |
| GCN normalization | Signed symmetric normalization precomputed per graph: `deg_i = s + Σ_j \|w_ij\|`, `ŵ_ij = w_ij / sqrt(deg_i·deg_j)` (self-loop gets `s/deg_i`). Passed as `edge_weight` to `GCNConv(normalize=False, add_self_loops=False)`. Degrees stay positive and signs are kept. |
| Node features | Learned node-ID embedding `nn.Embedding(num_nodes, embedding_dim)` |
| Demographics | `include_demo: true` by default (age, head motion → StandardScaler; sex, assessment centre → OneHot), fit on train |
| Target scaling | StandardScaler on train. Metrics reported on the original scale. |
| Sweep | keep ∈ {0.1, 0.2, …, 1.0} (10 levels), 1 seed (42) → 10 × 2 masks × 4 tests × 2 FC = **160 runs**. All of these are configurable. |
| Format | Python package + YAML config + CLI runner |

## File layout (all new, at repo root; nothing added to `others/`)
```
PLAN.md
requirements.txt                 # torch, torch_geometric, numpy, pandas, scikit-learn, pyyaml, joblib, matplotlib, tqdm
configs/topk.yaml
region_names/                    # copied from others/region_names, renamed (contents unchanged)
  FC21_regions.txt  FC21_network_groupings.json   # ← FC25_*
  FC55_regions.txt  FC55_network_groupings.json   # ← FC100_*
common/          # shared by every sparsification method (top-K now, RL later)
  __init__.py
  config.py      # load_config(path, overrides) -> dict; FC_TYPES / TARGETS maps
  data.py        # parse_region_map, load_subject_table, make_split, fit_preprocessors
  graphs.py      # signed_gcn_norm(...), build_graphs(...) -> list[Data]
  models.py      # GCNNet
  train.py       # train_epoch, evaluate, fit (early stopping, scheduler), get_optimizer_params
  utils.py       # set_deterministic_training, calc_r2_corr, EarlyStopping (adapted from others/)
topk/            # top-K edge-weight thresholding baseline
  __init__.py
  sparsify.py    # topk_edge_indices(scores, keep_frac) -> (N,k) idx   [plug point for the RL agent later]; shared_edge_indices, tile_mask
  run_sweep.py   # CLI over FC types × tests × keep levels × seeds
  summarize.py   # aggregate results.csv → table + R²-vs-keep% plots
tests/
  test_graphs.py        # shared: region maps, normalization, graph structure, model forward
  test_topk.py          # top-K selection
  test_topk_pipeline.py # end-to-end top-K sweep on a synthetic CSV (outputs, resume/skip)
```

## Component details

**config (`configs/topk.yaml`)**
```yaml
paths: {data_dir: ..., eid_dir: ..., region_dir: region_names, output_dir: outputs/topk}
fc_types:  {FC21: {regions_file: FC21_regions.txt, num_nodes: 21},
            FC55: {regions_file: FC55_regions.txt, num_nodes: 55}}
tests:     {GF: p20016_i2, PAL: p20197_i2, DSST: p23324_i2, TMT: p6350_i2}
demographics: {numeric: [p21003_i2, p25741_i2], categorical: [p31, p54_i2]}
split:     {val_frac: 0.15, test_frac: 0.15, seed: 42, use_eid_file: true}
graph:     {edge_value: standardized, self_loop_weight: 1.0}
sweep:     {keep_fracs: [0.1, 0.2, ..., 1.0], seeds: [42], masks: [personal, shared]}
model:     {embedding_dim: 32, hidden_channels: 64, gnn_num_layers: 2,
            classifier_hidden_dims: [32], node_dropout: 0.3, classifier_dropout: 0.3,
            pooling: attention, include_demo: true, demo_embed_dim: 8}
training:  {batch_size: 512, max_epochs: 200, lr: 0.005, weight_decay: 0.01,
            warmup_epochs: 10, early_stop_patience: 15, early_stop_min_delta: 0.0,
            compile: false, num_workers: 0}
```
The CLI can override any key (e.g. `--set training.lr=0.001`).

**data.py**
- `parse_region_map` is adapted from `others/process_graph_pipeline/csv_to_graph.py`. It returns `node_names`, `edge_list (E×2)`, and `columns` in the order of the file.
- `load_subject_table(test, fc_type, cfg)` reads the CSV (indexed by `eid`), optionally filters to the EID file, and checks for NaNs. It returns `X_raw (N×E float32)`, `X_demo_df`, `y`, `eids`.
- `make_split(eids, cfg)` returns train/val/test indices and writes `outputs/topk/splits/{TEST}_{train,val,test}_eids.txt`. The split depends only on the cognitive test, so FC21 and FC55 share it.
- `fit_preprocessors(...)` fits the demographics `ColumnTransformer`, the target `StandardScaler` and, only when needed, an edge `StandardScaler`, all on train. It saves them with joblib.

**topk/sparsify.py**: `topk_edge_indices(scores, keep_frac)`, where `scores = |raw X|`. When `keep_frac == 1.0` it returns all indices. The interface is a per-subject edge selection, so the RL agent can later supply its own selections in the same shape.

**graphs.py**
- `signed_gcn_norm(src, dst, w, num_nodes, s)` → `edge_index` (both directions plus self-loops) and normalized `edge_weight`, as defined in the decisions table.
- `build_graphs(X_val, top_idx, demo, y, eids, edge_list, num_nodes, cfg)` builds a list of `Data(x=arange(num_nodes), edge_index, edge_weight, demographics (1×D), y, eid)` Which edges were kept is not stored on each `Data` (to save memory); `edge_retention.npy` is computed from `top_idx` instead. The list is built in memory like the notebook's `BrainGraphDataset`, once per (FC, test, keep) and freed afterwards.

**models.py `GCNNet`** mirrors `GATv2ConvNet` from `others/gnn/architectures_normal.py`:
- node embedding (normal init, std=0.02)
- `gnn_num_layers` × `GCNConv(normalize=False, add_self_loops=False)`, each followed by BatchNorm → ELU → dropout except the last layer
- pooling: `AttentionalAggregation` by default, or mean
- concatenate `demo_projection(demographics)` when `include_demo`
- MLP head over `classifier_hidden_dims` (Linear → BN → ELU → Dropout) → `out_dim=1`
- Kaiming init on the custom Linear layers
- `forward(data)` uses `data.edge_weight`

**train.py** follows the notebook:
- AdamW with no weight decay on bias and norm parameters (`get_optimizer_params`)
- `LinearLR(start_factor=0.01, warmup_epochs)` → `CosineAnnealingLR`, combined with `SequentialLR`
- MSE loss on the scaled target
- early stopping on **val MSE** (patience 15), keeping the best `state_dict` on the CPU and restoring it at the end
- NaN or inf loss aborts the run and logs it as failed
- `evaluate()` inverse-transforms predictions and reports R², R²-corr, MAE and RMSE on train, val and test

**topk/run_sweep.py**
- CLI: `--config`, `--fc FC21 FC55`, `--tests GF PAL DSST TMT`, `--keep ...`, `--seeds ...`, `--max-subjects N` (for smoke tests), `--max-epochs`, `--overwrite`, `--set k=v`.
- Loop order: test → load the CSV once → FC type → preprocess once → keep level → seed.
- Each run writes to `outputs/topk/{FC}/{TEST}/{mask}/keep{pp}/seed{s}/` (shared runs also write `shared_edges.npy`):
  - `metrics.json`, `config.yaml` snapshot, `losses.csv` (train and val per epoch), `test_predictions.csv` (eid, actual, predicted), `model.pt` (state_dict)
  - `edge_retention.npy`: fraction of training subjects that kept each edge, for interpretability
- Each run appends one row to `outputs/topk/results.csv`: fc, test, keep_frac, k_edges, seed, n_train/val/test, best_epoch, epochs_run, train/val/test R², R²-corr, MAE, RMSE, minutes.
- Runs that already have `metrics.json` are skipped unless `--overwrite` is passed, so the sweep can resume.

**topk/summarize.py** reads `results.csv` and produces a pivot table (test R² by keep% per FC and test) and a 2×2 figure (one panel per test, R² vs keep%; colour = FC, solid = personal, dashed = shared), saved under `{paths.output_dir}/summary/`. It reads `paths.output_dir` from `--config` (default `configs/topk.yaml`, with `--set` overrides like `run_sweep.py`); `--results` / `--out-dir` still override.

## Verification
1. `pytest tests/test_graphs.py tests/test_topk.py` (local, synthetic data). The test checks:
   - the right number of edges is kept per subject (`k` undirected ⇒ `2k` directed + `num_nodes` self-loops)
   - kept edges have the largest |w|
   - signs are preserved
   - the edge set is symmetric
   - all normalized weights are finite, including for nodes that become isolated
   - `keep=1.0` gives the full graph
   - the GCNNet forward pass on a batch has output shape `(B,)`
2. Smoke run on the cluster: `python topk/run_sweep.py --fc FC21 --tests GF --keep 0.1 1.0 --max-subjects 2000 --max-epochs 3`. Confirm that the outputs are written, the results row is appended, and a rerun skips the finished runs.
3. Full FC21 GF at keep 1.0: check that test R² is in the same range as earlier GNN results for GF (a sanity check, not an exact match, since the architecture differs).
4. Full sweep (160 runs), then `topk/summarize.py` for the R²-vs-sparsity curves.

## Notes / risks
- Memory: FC55 at keep 1.0 holds ~3,000 directed edges per graph; with ~65k subjects that is a few GB in memory. That's fine on the cluster. If it becomes a problem, switch to building graphs lazily in `Dataset.__getitem__` from `top_idx`.
- Raw FC magnitudes are small relative to `self_loop_weight=1`. If the GCN ignores neighbours, try a smaller `self_loop_weight` in the config (it's a single setting, not a tuning sweep).

## Status (2026-10-28)
- [x] Implemented: `configs/`, `region_names/` (FC21/FC55 copies), `common/`, `topk/`, `tests/`
- [x] `pytest tests` passes locally (11 tests, synthetic data, CPU)
- [x] Renamed `brainsparse/` → shared `common/` + method-specific `topk/` (scripts and `configs/default.yaml` → `topk/`, `configs/topk.yaml`; outputs → `outputs/topk/`), so future methods (RL) sit beside `topk/` and reuse `common/`
- [x] Fast inference path for the RL stage: `GCNNet.forward_dense(adj, demo)` + `graphs.dense_gcn_adj` (batched torch signed norm, (B,N,N)) + `common/infer.py:predict_dense`. Same weights/maths as `forward` (equivalence tests in `tests/test_graphs.py`). Keep data resident on the GPU and pass (B,E) masks; no `Data`/`Batch`. ELU kept (negligible cost). `tests/bench_inference.py`: 6–9× faster than build+collate+sparse fwd on CPU; GPU numbers TBD
- [ ] Smoke run on the cluster with real UKB data (Verification step 2)
- [x] Shared-mask option (2026-10-31), 25 tests pass
- [ ] Full sweep (160 runs) + `topk/summarize.py`

## Usage
```bash
pip install -r requirements.txt
pytest tests                                                      # local sanity checks
python topk/run_sweep.py --fc FC21 --tests GF --keep 0.1 1.0 --max-subjects 2000 --max-epochs 3   # smoke
python topk/run_sweep.py                                       # full sweep from configs/topk.yaml
python topk/run_sweep.py --set graph.self_loop_weight=0.5 paths.output_dir=outputs/topk_sl05          # variant
python topk/summarize.py                                       # table + {output_dir}/summary/test_r2_vs_keep.png
python topk/summarize.py --set paths.output_dir=outputs/topk_sl05                                    # summarize a variant
```

## Change (2026-11-09): summarize.py reads paths from config
- `topk/summarize.py` now takes `--config` and `--set` and defaults `--results` to `{paths.output_dir}/results.csv` and `--out-dir` to `{paths.output_dir}/summary`. Previously it was hardcoded to `outputs/topk/`, which didn't match the config's output dir on scratch.

## Follow-up Q&A (2026-10-29), no code changes
- Demographic columns are listed in `configs/topk.yaml` (`demographics.numeric` / `categorical`) and read from the combined CSV. Nothing from `others/region_names` (demo_cols, col_renames data dictionary) needs to be copied.
- Targets are standardized on train only (`common/data.py`) and inverse-transformed before metrics (`common/train.py`).
- Within-subject FC normalization was considered and **declined**: no `graph.within_subject` option. Edges stay raw.

## Change (2026-10-31): group-standardized edge values
- Default switched to `graph.edge_value: standardized` in `configs/topk.yaml`. No code change was needed because `common/data.py:prepare_splits` already supported it. 18 tests pass.
- The scaler is fit on the dense train edges **before** top-K. That way one scaler serves every keep level, and dropped edges don't pull the mean and SD toward zero.
- Points to keep in mind:
  - The sign now means above or below the group mean, not positive or negative correlation.
  - An edge kept for having a large raw \|FC\| can have z ≈ 0, which makes it look the same as a dropped edge.
  - z-values are around 1 in size, much larger than raw FC, so neighbours carry more weight relative to `self_loop_weight=1`.
- The raw-value baseline was dropped at the user's request.
- RL stage: pass `prepared[split]['edge_vals']`, which holds the z-scores, to `predict_dense` and `dense_gcn_adj`, not `X_raw`.
- Shared mask added (see the Sparsification row). Why `frequency` is the default: it is the consensus of the personalized rule itself, so the comparison changes only whether the mask is personalized. `abs_mean` (|mean signed FC|) drops edges whose sign differs between subjects, which mixes sign consistency into the comparison. Averaging standardized edges is meaningless because every train mean is 0. At keep=1.0 both masks give the same graph and the same R², which serves as a sanity check.
- `prepare_splits` now also returns `rank_vals` (the signed ranking values). Run directories gained a `{mask}` level, and `results.csv` gained `mask` and `shared_score` columns. `summarize.py` treats old rows without `mask` as `personal`.

## Change (2026-11-01): one ranking rule per mask
- Removed `graph.rank_by` and `graph.shared_score`. **personal** always ranks by \|raw FC\|. **shared** always uses `frequency` (`shared_edge_indices(train_scores, keep_frac)`, which takes \|raw FC\|).
- Why \|raw FC\| for personal: it is the edge-weight thresholding baseline the proposal names. Ranking by \|z\| selects a subject's deviations from the group, which is a different idea and was incompatible with the shared mask anyway.
- Why `frequency` for shared: see the 2026-10-31 note. `mean_abs` picks nearly the same edges; `abs_mean` mixes sign consistency into the comparison.
- `prepare_splits` no longer returns `rank_vals`; `rank_scores` is always \|raw FC\|. The `shared_score` column was removed from `results.csv`, so move aside any existing `results.csv` written with that column before appending new runs (rows are appended without a header). 20 tests pass.
