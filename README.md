# csv-qa

### Overview
- **Environment ID**: `csv-qa`
- **Short description**: Synthetic CSV QA over a Faker-generated product catalog. Two single-turn tasks (sum per category, top-K expensive) with optional tool use (`csv_filter`, `csv_agg`)
- **Tags**: synthetic, tool-use, csv, tabular, arithmetic, hello-world

### Datasets
- **Primary dataset(s)**: Generated on-the-fly (Faker).
- **Split sizes**: N/A (custom-generated; TODO)

### Task
- **Type:** Single-turn prompts executed in **ToolEnv** (tools may be empty).
- **Parsers:** `XMLParser(fields=["answer"], answer_field="answer")`
- **Rubric overview:** `reward_exact_numeric_match` (parses `<answer>...</answer>` and compares to ground-truth). No ToolRubric used.

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval csv-qa
```

Defaults used in the published eval:

```bash
vf-eval csv_qa \
  --model gpt-4.1-mini \
  --num-examples 20 \
  --temperature 0.7 \
  --rollouts-per-example 3 \
  --save-dataset \
  --env-args '{
        "think": true,
        "allow_tool_use":true,
        "num_items":100,
        "num_categories":10,
        "locale":"en",
        "seed":42,
        "task_configs":{
            "sum_price_by_category":{},
            "top_k_expensive_by_category_sorted":{"top_k":3}
       }}'
```
These were not specifially tuned or anything.

<details> 

<summary>GPT-4.1-mini ablation</summary>

| Setting      | Think | Tool Use | Avg Reward (= Accuracy) | Std Dev | Correct / Total | Saved Dataset                                 |
| ------------ | ----- | -------- | ----------------------- | ------- | --------------- | --------------------------------------------- |
| GPT-4.1-mini | ✅     | ✅        | **0.950** (95.0%)       | 0.218   | 57 / 60         | `outputs/evals/csv_qa--gpt-4.1-mini/dd172a05` |
| GPT-4.1-mini | ❌     | ✅        | **0.217** (21.7%)       | 0.412   | 13 / 60         | `outputs/evals/csv_qa--gpt-4.1-mini/7fc45f47` |
| GPT-4.1-mini | ✅     | ❌        | **0.983** (98.3%)       | 0.128   | 59 / 60         | `outputs/evals/csv_qa--gpt-4.1-mini/d06eca6c` |
| GPT-4.1-mini | ❌     | ❌        | **0.167** (16.7%)       | 0.373   | 10 / 60         | `outputs/evals/csv_qa--gpt-4.1-mini/7fb58b6b` |

</details>


### Environment Arguments

| Arg              | Type | Default     | Description                                                                                            |
| ---------------- | ---- | ----------- | ------------------------------------------------------------------------------------------------------ |
| `think`          | bool | `true`      | If true, adds think prompt.                                                                            |
| `allow_tool_use` | bool | `true`      | If true, runs as `ToolEnv` with `csv_filter` & `csv_agg`, else behaves like single-turn.               |
| `num_items`      | int  | `100`       | Number of products generated.                                                                          |
| `num_categories` | int  | `10`        | Distinct categories.                                                                                   |
| `locale`         | str  | `"en"`      | Faker locale.                                                                                          |
| `seed`           | int  | `42`        | RNG/Faker seed for reproducibility.                                                                    |
| `task_configs`   | dict | see example | Adjust task configs. If a task is not provided, it is excluded from the dataset.                       |

**How many examples are evaluated on?**
With the current category-based tasks, for each included task, you get **one example per category**.
Total examples in the dataset = `num_categories * (#enabled_tasks)`.
The CLI `--num-examples` caps the total, which gets multiplied by `--rollouts-per-example`.


### Metrics

| Metric                       | Meaning                                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `reward`                     | The scalar reward (weighted sum across rubric fns). In this env it curr. equals `reward_exact_numeric_match`. |
| `reward_exact_numeric_match` | 1.0 iff the parser’s extracted `<answer>` exactly equals the ground-truth string; else 0.0.                   |

