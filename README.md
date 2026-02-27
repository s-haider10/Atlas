# ATLAS v2 — Compositional Safety Evaluation for AI Agents

ATLAS (Agent Testing through Layered Assessment of Safety) evaluates how AI agents handle **compound safety challenges** — scenarios where multiple safety dimensions interact simultaneously, as they do in real deployment.

Instead of testing safety dimensions in isolation, ATLAS composes scenarios from 4 source benchmarks using 5 composition operators, then measures whether models degrade when facing multiple safety pressures at once.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Source Benchmarks                        │
│  AgentHarm · InjecAgent · MoralStories · DarkBench          │
└────────────┬────────────────────────────────────────────────┘
             │ download + normalize
             ▼
┌─────────────────────────────────────────────────────────────┐
│  Atoms (standardized scenarios with tools + mock responses) │
└────────────┬────────────────────────────────────────────────┘
             │ compose (GPT-4o)
             ▼
┌─────────────────────────────────────────────────────────────┐
│  Tier 1: 140 atomic baselines (35 per benchmark)            │
│  Tier 2: 420 pairwise compositions (12 pairs × 35)          │
│  Tier 3: 210 triple compositions (6 configs × 35)           │
└────────────┬────────────────────────────────────────────────┘
             │ evaluate (Inspect AI)
             ▼
┌─────────────────────────────────────────────────────────────┐
│  Metrics: SIS · CDIM · CNI · RSI (with p-values + CIs)     │
│  Figures: heatmaps, violin plots, bar charts, scatter       │
└─────────────────────────────────────────────────────────────┘
```

**Composition operators:**
| Code | Operator | What it does |
|------|----------|-------------|
| IC | Instruction Conflict | Merges benign + adversarial instructions |
| TRC | Tool Response Contamination | Injects adversarial payload into tool outputs |
| NO | Norm Overlay | Layers ethical constraint onto a task |
| SE | Sycophancy Escalation | Adds social pressure to refusal scenarios |
| MDI | Moral Dilemma Injection | Embeds ethical dilemma into tool-use task |

**Metrics:**
| Metric | What it measures |
|--------|-----------------|
| SIS | Safety Interaction Score — per-dimension degradation from composition |
| CDIM | Cross-Dimension Interference Matrix — which dimensions hurt which |
| CNI | Composition Nonlinearity Index — Tier 3 vs Tier 2 compounding |
| RSI | Refusal Sensitivity Index — over-refusal vs genuine safety |

---

## Setup

```bash
# Clone and enter the repo
cd atlas

# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# Set up API keys
cp .env.example .env   # then edit with your keys
```

### Required `.env` file

```env
# Required: for scenario composition (GPT-4o) and evaluation
OPENAI_API_KEY=sk-...

# Optional: only needed if evaluating Anthropic models
ANTHROPIC_API_KEY=sk-ant-...

# Optional: only needed if evaluating via Together AI
TOGETHER_API_KEY=...

# Optional: only needed if evaluating Google models
GOOGLE_API_KEY=...
```

The `OPENAI_API_KEY` is always required because scenario composition and LLM judging use GPT-4o via the OpenAI client. Other keys are only needed for the specific model you want to evaluate.

---

## Quick Start — Full Pipeline

Run everything end-to-end with a single command:

```bash
uv run python run_pipeline.py --step all --model openai/gpt-4o-mini
```

This will: download benchmarks → normalize to atoms → build Tier 1/2/3 scenarios → validate → evaluate with the specified model → compute metrics and generate figures.

---

## Modular Pipeline — Step by Step

Each step can be run independently. Steps 1-6 only need to run once (they generate the scenario dataset). Steps 7-8 are run per model.

### Step 1: Download source benchmarks

```bash
uv run python run_pipeline.py --step download
```

Downloads AgentHarm, InjecAgent, MoralStories, and DarkBench to `raw_data/`. Takes ~3-5 minutes, ~500 MB disk.

### Step 2: Normalize to atoms

```bash
uv run python run_pipeline.py --step normalize
```

Converts raw benchmark data into standardized `Atom` objects saved to `processed/`.

### Step 3: Build Tier 1 (baselines)

```bash
uv run python run_pipeline.py --step tier1
```

Samples 35 atoms per benchmark (140 total) with stratified sampling. Enriches all tools with contextual mock responses. Output: `scenarios/tier1/tier1_scenarios.json`.

### Step 4: Build Tier 2 (pairwise compositions)

```bash
uv run python run_pipeline.py --step tier2
```

Uses GPT-4o to compose 12 benchmark pairs × 35 scenarios = 420 composed scenarios. Each scenario includes full tool definitions with mock responses generated in-context. Output: `scenarios/tier2/tier2_scenarios.json`.

**Requires `OPENAI_API_KEY`** — composition calls GPT-4o.

### Step 5: Build Tier 3 (triple compositions)

```bash
uv run python run_pipeline.py --step tier3
```

Chains a third benchmark onto Tier 2 scenarios: 6 configs × 35 = 210 triple-composed scenarios. Output: `scenarios/tier3/tier3_scenarios.json`.

### Step 6: Validate

```bash
uv run python run_pipeline.py --step validate
```

Runs structural validation on all scenario JSONs (checks required fields, tool definitions, dimension maps, etc.).

### Step 7: Evaluate a model

```bash
uv run python run_pipeline.py --step evaluate --model openai/gpt-4o-mini
```

Runs the model through all ~770 scenarios via Inspect AI. Logs are saved to `logs/` by Inspect.

You can evaluate a single tier:

```bash
uv run python run_pipeline.py --step evaluate --model openai/gpt-4o-mini --tier 2
```

### Step 8: Analyze results

```bash
uv run python run_pipeline.py --step analyze --model openai/gpt-4o-mini
```

Computes SIS, CDIM, CNI, RSI metrics with statistical tests (t-tests, chi-squared, Bonferroni correction). Generates figures to `figures/<model_name>/` — previous runs are never overwritten.

---

## Testing Multiple Models

Scenarios only need to be generated once (Steps 1-6). Then evaluate and analyze each model separately:

```bash
# Generate scenarios once
uv run python run_pipeline.py --step download
uv run python run_pipeline.py --step normalize
uv run python run_pipeline.py --step tier1
uv run python run_pipeline.py --step tier2
uv run python run_pipeline.py --step tier3
uv run python run_pipeline.py --step validate

# Evaluate each model (run one at a time)
uv run python run_pipeline.py --step evaluate --model openai/gpt-4o-mini
uv run python run_pipeline.py --step analyze  --model openai/gpt-4o-mini

uv run python run_pipeline.py --step evaluate --model openai/gpt-4o
uv run python run_pipeline.py --step analyze  --model openai/gpt-4o

uv run python run_pipeline.py --step evaluate --model anthropic/claude-sonnet-4-20250514
uv run python run_pipeline.py --step analyze  --model anthropic/claude-sonnet-4-20250514

uv run python run_pipeline.py --step evaluate --model together/meta-llama/Llama-3.1-70B-Instruct-Turbo
uv run python run_pipeline.py --step analyze  --model together/meta-llama/Llama-3.1-70B-Instruct-Turbo
```

Each analyze step saves figures to a separate folder:

```
figures/
├── openai_gpt-4o-mini/
│   ├── fig1_cdim_heatmap.png
│   ├── fig2_sis_distribution.png
│   ├── fig3_cni.png
│   └── fig4_rsi_vs_sis.png
├── anthropic_claude-sonnet-4-20250514/
│   └── ...
└── together_meta-llama_Llama-3.1-70B-Instruct-Turbo/
    └── ...
```

### Supported model providers

ATLAS uses [Inspect AI](https://inspect.ai-safety-institute.org.uk/) for evaluation, which supports many providers out of the box:

| Provider       | Model format                                                                                   | Required env var                                 |
| -------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| OpenAI         | `openai/gpt-4o-mini`, `openai/gpt-4o`, `openai/gpt-4.1`                                        | `OPENAI_API_KEY`                                 |
| Anthropic      | `anthropic/claude-sonnet-4-20250514`, `anthropic/claude-haiku-4-5-20251001`                    | `ANTHROPIC_API_KEY`                              |
| Together AI    | `together/meta-llama/Llama-3.1-70B-Instruct-Turbo`, `together/Qwen/Qwen2.5-72B-Instruct-Turbo` | `TOGETHER_API_KEY`                               |
| Google         | `google/gemini-2.0-flash`                                                                      | `GOOGLE_API_KEY`                                 |
| Ollama (local) | `ollama/llama3.1`, `ollama/qwen2.5`                                                            | None (local)                                     |
| Azure OpenAI   | `azureai/deployment-name`                                                                      | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` |

For Ollama (local models), make sure the Ollama server is running and the model is pulled first:

```bash
ollama pull llama3.1
uv run python run_pipeline.py --step evaluate --model ollama/llama3.1
```

### Running Inspect directly

You can also invoke Inspect AI directly for more control:

```bash
# Basic eval
inspect eval pipeline/to_inspect.py --model openai/gpt-4o-mini

# Specific tier only
inspect eval pipeline/to_inspect.py --model openai/gpt-4o-mini -T tier=2

# With Inspect's log viewer
inspect view
```

---

## Project Structure

```
atlas/
├── run_pipeline.py              # Main CLI entry point
├── download_all.py              # Downloads all 4 source benchmarks
├── requirements.txt             # Python dependencies
│
├── benchmarks/                  # Benchmark downloaders
│   ├── download_agentharm.py
│   ├── download_injecagent.py
│   ├── download_moral_stories.py
│   └── download_darkbench.py
│
├── processing/                  # Raw → Atom normalization
│   └── normalize.py
│
├── schemas/                     # Data models
│   └── atom.py                  # Atom + ToolDefinition dataclasses
│
├── composition/                 # Scenario composition
│   ├── operators.py             # 5 composition operators (IC/TRC/NO/SE/MDI)
│   ├── build_tier1.py           # Tier 1 baseline sampling
│   ├── build_tier2.py           # Tier 2 pairwise composition
│   ├── build_tier3.py           # Tier 3 triple composition
│   └── validate.py              # Schema validation
│
├── tools/                       # Tool mock system
│   └── tool_registry.py         # Mock response generation + Inspect Tool builders
│
├── configs/                     # Configuration
│   └── api.py                   # Model names, OpenAI client
│
├── pipeline/                    # Inspect AI integration
│   └── to_inspect.py            # Scenario → Inspect Sample + Task
│
├── scorers/                     # Evaluation scorers
│   ├── multi_scorer.py          # Per-dimension scoring with tool audit
│   ├── agentharm_scorer.py      # AgentHarm-specific scorer
│   ├── injecagent_scorer.py     # InjecAgent injection detection
│   ├── darkbench_scorer.py      # DarkBench pattern scorer
│   └── moral_stories_scorer.py  # MoralStories norm compliance
│
├── analysis/                    # Metrics computation
│   ├── sis_computation.py       # Safety Interaction Score
│   ├── cdim_matrix.py           # Cross-Dimension Interference Matrix
│   ├── cni_computation.py       # Composition Nonlinearity Index
│   ├── rsi_computation.py       # Refusal Sensitivity Index
│   ├── stats_utils.py           # t-tests, CIs, Bonferroni
│   ├── utils.py                 # Eval log loading
│   └── visualization.py         # Figure generation
│
├── scenarios/                   # Generated scenario JSONs (gitignored)
│   ├── tier1/
│   ├── tier2/
│   └── tier3/
│
├── figures/                     # Generated figures (gitignored)
│   └── <model_name>/            # Per-model figure folders
│
├── logs/                        # Inspect eval logs (gitignored)
├── raw_data/                    # Downloaded benchmarks (gitignored)
└── processed/                   # Normalized atoms (gitignored)
```

---

## Tool Mock Pipeline

All tool mock responses are generated at **composition time** (not eval time), ensuring:

1. **Reproducibility** — mock responses are stored in scenario JSONs, identical across eval runs
2. **Context-awareness** — mocks are generated by GPT-4o using the scenario's user prompt, safe/unsafe behaviors
3. **Audit trail** — every scenario carries a `tool_audit` manifest listing each tool's name, description, response type (`mock`/`injected`/`heuristic`), and exact response. The LLM judge receives this context when scoring.

```
operators.py (LLM generates tool dicts with mock_response)
    ↓
build_tier{1,2,3}.py (generate_mock_responses() fills gaps)
    ↓
scenarios/*.json (tools stored as full dicts)
    ↓
to_inspect.py (tools + tool_audit → Sample.metadata)
    ↓
inject_scenario_tools() (builds Inspect Tool objects)
    ↓
multi_scorer.py (tool_audit in judge prompt + Score.metadata)
```

---

## Statistical Methodology

All metrics include proper statistical tests:

- **SIS**: One-sample t-test (H0: SIS = 0), Bonferroni-corrected p-values, 95% CIs, Cohen's d
- **CDIM**: Per-cell t-tests with sample counts
- **CNI**: One-sample t-test (H0: CNI = 0) per triple config
- **RSI**: Chi-squared / Fisher's exact test comparing refusal rates (Tier 1 vs Tier 2)

Sample sizes (n=35 per group) are powered for medium effect sizes (Cohen's d = 0.5) at alpha = 0.05, power = 0.80.
