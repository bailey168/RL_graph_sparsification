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
| Split | 70/15/15 train/val/test via `train_test_split`, as in the notebook. **Per-seed split (2026-11-10):** split seed = training seed (`split.per_seed: true`), and each seed's split is shared by every keep level, mask and FC type. Split EIDs are saved to `splits/seed{s}/`. **The RL stage uses the seed-0 split.** Missing values raise an error (notebook behaviour). |
| Sparsification | Two masks are compared (`sweep.masks`). **personal**: each subject keeps `k = max(1, floor(p·E))` edges (`k = 0` at keep 0.0, which leaves self-loops only: the demographics-only reference) with the largest **\|raw FC\|** (`np.argpartition`). **shared** (added 2026-10-31): one set of k edges from the train split, applied to every subject. Edges are ranked by **frequency**: how often an edge is in a train subject's personal top-K, with ties broken by mean \|FC\|. One rule per mask, no ranking options (simplified 2026-11-01). |
| Edge value | **Group z-score** of each edge (`edge_value: standardized`, changed 2026-10-31). A `StandardScaler` per edge is fit on the **pre-sparsified** train edges, then applied to all splits. Selection always ranks by raw \|FC\|. Setting `edge_value: raw` still works but is no longer a baseline we run. |
| Self-loops | Kept, with `self_loop_weight` (default 1.0) |
| GCN normalization | Signed symmetric normalization precomputed per graph: `deg_i = s + Σ_j \|w_ij\|`, `ŵ_ij = w_ij / sqrt(deg_i·deg_j)` (self-loop gets `s/deg_i`). Passed as `edge_weight` to `GCNConv(normalize=False, add_self_loops=False)`. Degrees stay positive and signs are kept. |
| Node features | Learned node-ID embedding `nn.Embedding(num_nodes, embedding_dim)` |
| Demographics | `include_demo: true` by default (age, head motion → StandardScaler; sex, assessment centre → OneHot), fit on train |
| Target scaling | StandardScaler on train. Metrics reported on the original scale. |
| Sweep | keep ∈ {0.0, 0.1, …, 1.0} (11 levels), seeds 0–9. Shared is skipped at 0.0 and 1.0, where it gives the same graphs as personal → 10 × (9×2 + 2) × 4 tests × 2 FC = **1,600 runs** (2026-11-10; was 1 seed, 160 runs). All of these are configurable. |
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
  models.py      # GCNNet, GATv2Net (model.type), build_model, graph_format
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
- [x] Single-seed sweep (160 runs, seed 42, fixed split). The curves were too noisy to interpret, which led to the multi-seed sweep below
- [x] Multi-seed code (2026-11-10), 23 tests pass
- [ ] Move the old seed-42 output folder aside by hand, then run the full 1,600-run sweep + `topk/summarize.py`
- [x] Optuna-tuned sweep code (2026-11-19, see below), 28 tests pass
- [x] Optuna scope narrowed to FC55 × {GF, DSST}, pooling fixed, edge dropout added, auto-summary (2026-11-19), 29 tests pass
- [ ] Optuna cluster smoke test, then the full 40-study run (summary is written automatically)
- [x] GATv2 model option + `configs/topk_gat_optuna.yaml` (2026-11-21, see below), 43 tests pass
- [ ] GATv2 Optuna: cluster smoke test, then GF keep 1.0 (compare with the GCN study), then the full 40 studies
- [x] Faster training with identical results (2026-11-21, see below): device-resident tensor loader, cheaper epoch loop, file-locked CSV appends for parallel studies
- [ ] Speed check on a cluster GPU: `python tests/bench_training.py --device cuda`, then a 3-trial smoke study with `--set training.loader=pyg` vs the default, diffing trials.csv/metrics.json

## Usage
```bash
pip install -r requirements.txt
pytest tests                                                      # local sanity checks
python topk/run_sweep.py --fc FC21 --tests GF --keep 0.1 1.0 --max-subjects 2000 --max-epochs 3   # smoke
python topk/run_sweep.py                                       # full sweep from configs/topk.yaml
python topk/run_sweep.py --set graph.self_loop_weight=0.5 paths.output_dir=outputs/topk_sl05          # variant
python topk/summarize.py                                       # mean ± SD + paired Δ tables/figures in {output_dir}/summary/
python topk/summarize.py --set paths.output_dir=outputs/topk_sl05                                    # summarize a variant
# Optuna-tuned sweep (one study per FC × test × mask × keep); defaults FC55 × {GF, DSST} from sweep.fc/sweep.tests
python topk/run_optuna.py --tests GF --keep 0.1 1.0 --max-subjects 2000 --n-trials 5 --max-epochs 5   # smoke
python topk/run_optuna.py --tests GF                           # one cluster job per test; resumable; summarizes at the end
python topk/summarize.py --config configs/topk_optuna.yaml     # re-summarize by hand (e.g. after parallel jobs)
# GATv2 instead of GCN: same scope and search space + heads, separate output dir (topk_gat_optuna)
python topk/run_optuna.py --config configs/topk_gat_optuna.yaml --tests GF --keep 0.0 1.0 --max-subjects 2000 --n-trials 3 --max-epochs 5   # smoke
python topk/run_optuna.py --config configs/topk_gat_optuna.yaml --tests GF --keep 1.0   # first real check vs GCN
python topk/run_sweep.py --set model.type=gatv2 model.heads=4 paths.output_dir=...      # fixed-hyperparameter GAT sweep
```

## Review (2026-11-21): new top-K/Optuna code vs old GAT notebook, no code changes
Question: why the Optuna top-K results show no clear R²-vs-keep pattern and are worse than the old GAT.
- **Matches the old notebook** (`GNN_all_scaling_law_optuna_nocv.ipynb`): 70/15/15 split, train-only scalers (edges z-scored before top-K), top-K by |raw FC| with `k = max(1, floor(p·E))` and z-values on the edges, AdamW with no decay on bias/norm, warmup + cosine, early stopping on val MSE, best-state restore. No bugs found.
- **Differs (likely cause of the gap):**
  1. Model: old = `GATv2ConvNet_Inject` (`edge_attr` enters the attention logits only, `residual=True`, GATv2 adds self-loops, demographics injected into node features between conv layers, no demographics at readout). New = `GCNConv` with signed sym-normalized z-scores *multiplying* messages and demographics concatenated at readout.
  2. Search space: old lr 1e-3–3e-2, hidden 32–128, embedding ≥ 32, 2–4 conv layers, 100 trials. New lr 1e-4–1e-2, hidden 16–256, embedding ≥ 8, 1–4 layers, 50 trials.
  3. The old "fewer edges → worse" impression came from `sparsity_percent` (0.3–1.0) being *one tuned hyperparameter inside a single study* (`GNN_all_scaling_law_optuna.ipynb`), which mixes keep level with the other params and with TPE sampling. It was not a controlled comparison. The notebooks have no saved outputs, so the old numbers can't be checked here.
- Suggested next checks (not done): Δ test R² vs keep 0.0 (if ≈ 0, the GCN isn't using FC); a ridge baseline on masked edges + demographics per keep level; a GATv2 option in `common/models.py` run through the same pipeline at keep 1.0 to separate the model's effect from the pipeline's.

## Change (2026-11-21): GATv2 model option + Optuna-tuned GATv2 sweep
Why: the review above found the model is the main difference from the old GAT notebook. GATv2 now runs through the same top-K + Optuna pipeline, so GCN and GAT are compared on identical splits, masks and search ranges.

| Topic | Decision |
|---|---|
| Architecture | `GATv2Net` in `common/models.py`: `GATv2ConvNet` from `others/gnn/architectures_normal.py` **without inject or jumping knowledge**. Node-ID embedding → `gnn_num_layers` × `GATv2Conv(edge_dim=1, residual=True, concat=False, heads=H, add_self_loops=True, fill_value='mean')` → BatchNorm → ELU → dropout (not after the last layer) → attention pooling → + demographics projection → MLP head. |
| Selection | `model.type: gcn \| gatv2` (default `gcn`). `build_model` dispatches on it, and `graph_format(model_cfg)` tells `build_graphs` which edge format to build. |
| Edge input | `graph_format='edge_attr'`: kept **z-scored** edge values as `edge_attr` (2k × 1), both directions, no self-loops, no GCN normalization. They enter the attention logits only. GATv2Conv adds self-loops whose attr is the mean incoming attr (0 for isolated nodes and everywhere at keep 0.0). `graph.self_loop_weight` is unused for GAT. |
| `edge_dropout` | GATv2 **attention dropout** (`GATv2Conv(dropout=...)`), as in the reference. Unlike GCN's DropEdge, it can also drop self-loop attention. |
| Search space | Inherited GCN space **plus** `model.heads` (pow2 1–4 → 2–16). No hidden-size constraint because heads are averaged. |
| RL dense path | Deferred: `GATv2Net` has no `forward_dense`, and `predict_dense` raises `NotImplementedError` for it. |
| Scope | `configs/topk_gat_optuna.yaml` (`base: topk_optuna.yaml`): same FC55 × {GF, DSST} × 20 masks/keeps = 40 studies, 50 trials, seed 0, output `/scratch/bng/cartbind/data/RL_outputs/topk_gat_optuna`. |

Files:
- `common/models.py`: shared `_GraphNet` base (embedding, norms, pooling, demographics, head, init, `_post_conv`, `_readout`); `GCNNet` and `GATv2Net` supply the conv layer. GCN parameter names, module order and initialization are unchanged: the state_dict at a fixed seed is bit-identical to the old code, so old `model.pt` files load.
- `common/graphs.py`: `build_graphs(..., graph_format='gcn_norm' | 'edge_attr')`.
- `topk/run_sweep.py` / `run_optuna.py`: `build_split_graphs(..., graph_format)` is fed from `graph_format(cfg['model'])`, so the fixed sweep can also run GAT (`--set model.type=gatv2`).
- `configs/topk.yaml`: + `model.type: gcn`, `model.heads: 4`. New `configs/topk_gat_optuna.yaml`. `common/infer.py`: clear error for models without `forward_dense`.
- Tests: edge_attr graph structure (keep 0.0 → empty), GATv2 forward/backward (incl. keep 0.0), sensitivity to edge values, attention dropout off in eval, `build_model` dispatch, and an end-to-end GAT Optuna run (`tests/test_topk_optuna_pipeline.py::test_gat_optuna_sweep`). 43 tests pass.

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

## Change (2026-11-10): multi-seed sweep with a split per seed
Why: the single-seed curves were jagged. Jumps between neighbouring keep levels (e.g. PAL FC21 shared 0.080 → 0.090 → 0.085) were as large as the effects of interest. The pipeline is deterministic (personal = shared at 100%), so this was training and sampling variance.
- **Seeds 0–9** for every config. The old seed-42 runs are not reused; the user moves the old output folder aside by hand before running (`paths.output_dir` is unchanged).
- **Split per seed** (`split.per_seed: true`). Error bars then cover both training randomness and which subjects are sampled. Within a seed all configs share the split, so comparisons stay paired. Preprocessors and the shared mask are refit per seed on that seed's train split. `split.seed` is still used for `--max-subjects` subsampling and as the split seed when `per_seed: false`.
- **Keep 0.0** added: `num_kept_edges` returns 0, so graphs have self-loops only (normalized weight 1). The graph branch gives the same vector for every subject, which makes the model effectively demographics-only. Fixed two k = 0 edge cases in `topk/sparsify.py`: `[:, -0:]` and `order[-0:]` would have selected every edge.
- **Shared skipped at 0.0 and 1.0** (`topk.sparsify.is_redundant`). `summarize.py` copies the personal rows into those points.
- `run_sweep.py` loop: test → CSV → split seed → FC → preprocess → (keep, mask, seed). Outputs: `splits/seed{s}/{TEST}_{split}_eids.txt`, `{FC}/{TEST}/preprocessors_seed{s}.pkl`. `results.csv` and `metrics.json` gained a `split_seed` column; run directories are unchanged.
- `summarize.py`:
  - `{metric}_vs_keep.png` shows the mean with **±1 SD error bars** across seeds, with personal and shared slightly x-offset.
  - New **paired Δ vs 100% kept**: per seed, the metric minus the personal keep-1.0 run on the same split, aggregated as mean ± SD (`{metric}_delta_vs_full.csv/.png`).
  - Prints a warning if any config has fewer seeds than the rest.
- Tests: keep-0 graphs, `is_redundant`, per-seed splits and skipping in the pipeline test (now `--seeds 0 1`), summarize run end-to-end, and a new `tests/test_summarize.py`.


## Change (2026-11-19): Optuna-tuned sweep (one tuned model per sparsity level)
Why: one fixed hyperparameter config can suit some keep levels better than others, which mixes that fit into the R²-vs-sparsity curve. Each sparsity level now gets its own hyperparameter search and one final model, and the 10-seed repeats are dropped for this run. The 10-seed sweep code and outputs are left as they were.

| Topic | Decision |
|---|---|
| Studies | One per (FC, test, mask, keep): 4 × 2 × (11 personal + 9 shared) = **160 studies**. Shared is still skipped at 0.0 and 1.0. |
| Trials | **100 per study**: TPE (`multivariate`, `group`, 10 random startup trials). Pruner is **off by default** (`optuna.pruner.enabled: false`, updated 2026-11-19): every trial trains to early stopping. Turn on with `--set optuna.pruner.enabled=true`, which uses Hyperband (`min_resource` 50, `reduction_factor` 2, `max_resource` = `max_epochs`). |
| Objective | Best val MSE (scaled target) from early stopping. Test metrics are stored as trial user attrs only and never used for selection. |
| Seed | **0** for the split, every trial's training and the sampler. The RL stage uses the same split. Trials differ only in their hyperparameters. Because the sampler seed is the same, every study starts with the same 10 random configs, which also makes studies easier to compare. |
| Final model | The best trial's weights, saved when the trial finishes. No retraining, unless the checkpoint on disk doesn't match `study.best_trial` after a crash; then that trial is retrained with the same seed. |
| Search space | (updated 2026-11-19) `hidden_channels` 16–256, `embedding_dim` 8–min(64, hidden) (`at_most: model.hidden_channels`, inclusive), `gnn_num_layers` 1–4, `classifier_hidden_dims` (1–3 strictly decreasing powers of 2, first layer ≤ hidden/2 via strict `below:`, last ≥ 4), `node_dropout` / `classifier_dropout` / `edge_dropout` 0–0.7, `demo_embed_dim` 2–8, `lr` 1e-4–1e-2 (log), `weight_decay` 1e-5–1e-1 (log). **Fixed:** `pooling` (attention), `self_loop_weight` (it interacts with sparsity), `batch_size`, `warmup_epochs`, `max_epochs`, patience, `edge_value`. |
| Output | `/scratch/bng/cartbind/data/RL_outputs/topk_optuna` |

Files:
- `common/tuning.py` (new, method-agnostic, for reuse by RL):
  - config-driven search space (`int`, `float`, `categorical`, `pow2`, `decreasing_pow2`; dynamic upper bounds `at_most:` inclusive / `below:` strict from an earlier key), `suggest_config` and `config_from_params`
  - `make_study` (one SQLite file per study, resumable), `remaining_trials`, `pruning_callback`, `is_new_best`
  - `save_study_outputs`: `trials.csv`, `best_params.json`, `param_importances.json` (fANOVA), and plots of optimization history, parallel coordinates, param importances and slices. Each plot is saved as HTML (plotly) and PNG (matplotlib backend, so no kaleido/Chrome is needed on the cluster).
- `common/config.py`: a `base: other.yaml` key deep-merges a parent config. `configs/topk_optuna.yaml` inherits `topk.yaml` this way.
- `common/train.py`: `fit(..., epoch_callback=None)` is used for pruning, and the result gains `best_val_loss`. The existing sweep's behaviour is unchanged.
- `topk/run_optuna.py`: graphs are built once per study.
  - Per study it writes `{output_dir}/{FC}/{TEST}/{mask}/keep{pp}/`, containing `study.db`, `trials.csv`, `best_params.json`, `param_importances.json`, `config.yaml` (with the best params applied), `model.pt`, `losses.csv`, `test_predictions.csv`, `best_trial.json`, `edge_retention.npy`, `[shared_edges.npy]`, `plots/` and `metrics.json` (written last).
  - It appends to `results.csv`, which has the same columns as the old sweep (seed 0) plus `best_trial`, `n_trials_complete`, `n_pruned` and `n_failed`, and to `best_params.csv`.
  - A study without `metrics.json` resumes and tops up to `n_trials`. `--overwrite` restarts a study from scratch.
- `topk/summarize.py`: handles one seed (no error bars; labelled "single seed"). If the Optuna files exist, it also writes `best_params_vs_keep_{TEST}.png` and `param_importance_by_keep.csv` / `param_importance_by_keep_{FC}_{mask}.png`.
- `requirements.txt`: + optuna, plotly. Tests: `tests/test_tuning.py`, `tests/test_topk_optuna_pipeline.py`.
- RL later: write `configs/rl_optuna.yaml` with its own `optuna.search_space` and objective, and reuse `common/tuning.py`.

## Change (2026-11-19, later): narrower Optuna scope, edge dropout, auto-summary
- **Scope:** only **FC55** and **GF, DSST** → 2 × (11 + 9) = **40 studies**. Set with new optional keys `sweep.fc` / `sweep.tests` in `configs/topk_optuna.yaml`; `--fc` / `--tests` still override. `fc_types` / `tests` maps in `topk.yaml` stay complete (the loaders need them).
- **Pooling** is fixed to attention and no longer searched.
- `configs/topk_optuna.yaml` now states `sweep.keep_fracs` and `sweep.masks: [personal, shared]` explicitly instead of inheriting them from `topk.yaml` (same values), so the run's full scope is visible in one file.
- **Edge dropout (DropEdge)** was added to `GCNNet` (`model.edge_dropout`, default 0.0 in `topk.yaml`, so the fixed sweep is unchanged) and is searched over 0–0.5 (step 0.05).
  - During training, each non-self-loop entry of the precomputed normalized `edge_weight` is zeroed with probability p, and the survivors are scaled by 1/(1-p) (inverted dropout, no renormalization). Self-loops are always kept, and one mask is shared by all conv layers. `forward_dense` does the same on the off-diagonal of `adj`.
  - Eval mode is unaffected, so the RL inference path and the sparse/dense equivalence are unchanged.
  - Why: the dense model gets its own regularizer against noisy edges, which makes the comparison with top-K fairer, and the model may be more robust when scoring RL-sparsified graphs. It does nothing at keep 0.0.
- **Auto-summary:** `run_optuna.py` and `run_sweep.py` call `topk.summarize.auto_summarize(output_dir)` at the end (`--no-summary` to skip). It writes into `{output_dir}/summary/` and never raises. `summarize.py`'s logic now lives in `summarize(results_path, out_dir, metric)`. With parallel jobs, the last one to finish writes the complete summary, and it can also be rerun by hand.

## Change (2026-11-19): simplification refactor (no behaviour change)
- `topk/run_sweep.py` now holds shared helpers that `run_optuna.py` reuses: `keep_dir`, `build_split_graphs`, `prepare_fc`, `save_history`, `save_mask`, `result_row`, `write_json` and `free_memory`. Results columns and order are unchanged.
- `get_dotted` / `set_dotted` moved from `common/tuning.py` into `common/config.py` (also used for `--set` overrides).
- `GCNNet`: `_post_conv` and `_readout` are shared by `forward` and `forward_dense`. Parameter names and init are unchanged, so old `model.pt` files still load.
- Smaller cleanups in `common/data.py`, `common/train.py` and `topk/summarize.py` (`save_fig`, `fc_colors`, `LINESTYLES`).
- Tests: `tests/conftest.py` holds the shared `fc21` fixture, and `pytest.ini` sets `pythonpath = .` (no more `sys.path` hacks in test files).
- Verified: all 31 tests pass. Synthetic `run_sweep.py` and `run_optuna.py` outputs (results.csv, metrics.json, model.pt tensors, splits, masks, Optuna trials) are identical to the pre-refactor code, apart from timing fields.

## Change (2026-11-21): faster training, bit-identical results
Goal: speed up the Optuna GATv2 runs (one GPU per job) **without changing any result**. These would change results and are **not** done: turning off deterministic algorithms, AMP/TF32, `torch.compile`, parallel trials within one study (that changes the TPE sample sequence), the pruner.
- **`common/loader.py`** (new): `GraphTensors` stacks a split's `Data` list (from `build_graphs`, so the values are the same tensors) on the device. `TensorLoader` assembles each minibatch with a few indexing ops (node offsets as in `Batch.from_data_list`), with no PyG Python collation per batch and no host→device copy. The shuffle order comes from a torch `DataLoader(range(n), …, generator)`, so the sampler and generator draws match the PyG loader exactly. It needs equal edge counts per graph (always true for top-K).
- New config key **`training.loader: tensor | pyg`** (default `tensor`). `run_sweep.stage_graphs` builds the tensors once per run/study; `run_optuna` shares them across all trials.
- `common/train.py`: train and val losses are summed as a float64 device tensor (same IEEE ops/order as the old Python-float sum), with one host sync per epoch. The per-epoch val pass uses the new `eval_loss` (no CPU copies of predictions). `EarlyStopping` keeps its best state on the device.
- `append_row` holds an `fcntl` lock, so several processes can append to the same results.csv / best_params.csv / failures.csv. **Parallel studies on one GPU** (exact, since studies are independent): `for k in 1.0 0.8 0.6; do python topk/run_optuna.py --config configs/topk_gat_optuna.yaml --keep $k --no-summary & done; wait; python topk/summarize.py --config configs/topk_gat_optuna.yaml`.
- Verified: `tests/test_loader.py` (every batch field equal to the PyG `Batch` over 2 shuffled epochs, GCN/GATv2 × keep 0/0.5/1.0 × personal/shared; `fit()` with `pyg` vs `tensor` gives identical history, metrics and weights). A one-off script also showed `fit()` matches the **original HEAD** `train.py`/`utils.py` + PyG loader bit for bit (GCN/GATv2, keep 0/0.3/1.0).
- Speed: on a laptop **CPU**, GATv2 epochs are dominated by the model itself (FC21, 4,200 train subjects: GATv2 14.8 → 14.2 s/epoch, GCN 1.31 → 1.27 s/epoch), so the loader gives little there. The gain is expected on the **GPU**, where the per-batch CPU collation was the bottleneck; to be measured with `tests/bench_training.py --device cuda`.
