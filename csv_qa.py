import abc
import csv
import io
import json
import random
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from textwrap import indent
from typing import Callable, Literal

import verifiers as vf
from datasets import Dataset
from faker import Faker


# Data generation


@dataclass(frozen=True, slots=True)
class Product:
    id: int
    name: str
    category: str
    price: Decimal


def create_catalog(n: int, n_categories: int, seed: int, locale: str = "en") -> tuple[str, list[Product]]:
    rng = random.Random(seed)
    fake = Faker(locale=locale)
    Faker.seed(seed)
    words = fake.words((n * 2) + n_categories, unique=True)
    categories = words[:n_categories]
    rows: list[Product] = [
        Product(
            id=i,
            name=" ".join(words[(i * 2) - 1 + n_categories : (i * 2) + 1 + n_categories]).title(),
            category=rng.choice(categories),
            price=Decimal(f"{rng.uniform(1.50, 99.99):.2f}"),
        )
        for i in range(1, n + 1)
    ]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "name", "category", "price"])
    for r in rows:
        w.writerow([r.id, r.name, r.category, f"{r.price:.2f}"])
    return buf.getvalue(), rows


# Optional tools


def csv_filter(csv_data: str, category: str, top_k: int) -> str:
    """
    Filter rows in the CSV by 'category' and return the top_k most expensive items.
    Returns JSON: {"category": str, "top_k": int, "items": [{"id": int, "name": str, "category": str, "price": "<two-decimal str>"}]}.
    """
    items = [
        {"id": int(r["id"]), "name": r["name"], "category": r["category"], "price": f"{Decimal(r['price']):.2f}"}
        for r in csv.DictReader(io.StringIO(csv_data))
        if r["category"] == category
    ]
    items.sort(key=lambda x: Decimal(x["price"]), reverse=True)
    return json.dumps({"category": category, "top_k": top_k, "items": items[:top_k]})


def csv_agg(csv_data: str, op: Literal["sum", "mean", "median"]) -> str:
    """
    Compute an aggregate over 'price' in the CSV.
    Returns JSON: {"op": str, "value": "<two-decimal str>"}.
    """
    prices: list[Decimal] = [Decimal(r["price"]) for r in csv.DictReader(io.StringIO(csv_data))]
    if not prices:
        val = Decimal("0.00")
    elif op == "sum":
        val = sum(prices)
    elif op == "mean":
        val = sum(prices) / len(prices)
    elif op == "median":
        ps = sorted(prices)
        n = len(ps)
        mid = n // 2
        val = (ps[mid - 1] + ps[mid]) / 2 if n % 2 == 0 else ps[mid]
    else:
        return json.dumps({"error": f"invalid op {op}"})
    return json.dumps({"op": op, "value": f"{val:.2f}"})


# Tasks


class Task(abc.ABC):
    name: str
    prompt: str
    parser: vf.Parser

    @abc.abstractmethod
    def build_dataset(self, data: list[Product], csv_data: str) -> Dataset:
        """Create a dataset with 'question' and 'answer' fields."""

    @abc.abstractmethod
    def rubric_group(self) -> vf.RubricGroup: ...


class SumPriceByCategory(Task):
    name = "sum_price_by_category"
    prompt = """
<task>
    Return the total price of all products in the specified category from the provided CSV data.
</task>
<format>
    The answer should be the exact sum formatted to two decimals, enclosed in an "answer" XML tag. Do not include any other text in this answer tag.
</format>
<example id="1">
    <user>
        <csv category="Widgets">
            id,name,category,price
            1,Widget A,Tools,19.99
            2,Gadget B,Electronics,29.99
            3,Widget C,Tools,9.99
        </csv>
    </user>
    <assistant>
        <answer>29.98</answer>
    </assistant>
</example>
"""
    parser = vf.XMLParser(fields=["think", "answer"], answer_field="answer")

    @staticmethod
    def create_task(data: list[Product], csv_data: str, category: str) -> tuple[str, str]:
        question = f"""
<csv category="{category}">
{indent(csv_data.strip(), " " * 4)}
</csv>
"""
        total = sum(p.price for p in data if p.category == category)
        return question, f"{total:.2f}"

    def build_dataset(self, data: list[Product], csv_data: str) -> Dataset:
        return build_category_task_dataset(
            partial(self.create_task, data=data, csv_data=csv_data), data=data, task_name=self.name
        )

    def rubric_group(self) -> vf.RubricGroup:
        # NOTE this only works if the parser is in the RubricGroup. But then how to have multiple rubrics with different parsers?
        exact_match_rubric = vf.Rubric(
            funcs=[reward_exact_numeric_match],
            weights=[1.0],
            parallelize_scoring=True,
        )
        return vf.RubricGroup(rubrics=[exact_match_rubric], parser=self.parser)


class TopKExpensiveByCategorySorted(Task):
    name = "top_k_expensive_by_category_sorted"
    prompt = """
<task>
    Return the top K most expensive products in the specified category from the provided CSV data.
</task>
<format>
    The answer should be a JSON array of product names, sorted by price in descending order, enclosed in an "answer" XML tag. Do not include any other text in this answer tag.
</format>
<example id="1">
    <user>
        <csv category="Tools" top_k="2">
            id,name,category,price
            1,Widget A,Tools,19.99
            2,Gadget B,Electronics,29.99
            3,Widget C,Tools,9.99
            4,Widget D,Tools,39.99
        </csv>
    </user>
    <assistant>
        <answer>["Widget D", "Widget A"]
    </assistant>
</example>
"""
    parser = vf.XMLParser(fields=["think", "answer"], answer_field="answer")

    def __init__(self, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k

    def create_task(self, data: list[Product], csv_data: str, category: str) -> tuple[str, str]:
        question = f"""
<csv category="{category}" top_k="{self.top_k}"> 
{indent(csv_data.strip(), " " * 4)}
</csv>
"""
        top_item_names = [
            p.name
            for p in sorted((p for p in data if p.category == category), key=lambda x: x.price, reverse=True)[
                : self.top_k
            ]
        ]
        return question, json.dumps(top_item_names)

    def build_dataset(self, data: list[Product], csv_data: str) -> Dataset:
        return build_category_task_dataset(
            partial(self.create_task, data=data, csv_data=csv_data), data=data, task_name=self.name
        )

    def rubric_group(self) -> vf.RubricGroup:
        # NOTE this only works if the parser is in the RubricGroup. But then how to have multiple rubrics with different parsers?
        # https://github.com/willccbb/verifiers/issues/327
        exact_match_rubric = vf.Rubric(
            funcs=[reward_exact_numeric_match],
            weights=[1.0],
            parallelize_scoring=True,
        )
        return vf.RubricGroup(rubrics=[exact_match_rubric], parser=self.parser)


def build_category_task_dataset(
    create_task_fn: Callable[[str], tuple[str, str]],
    data: list[Product],
    task_name: str,
) -> Dataset:
    tasks = [
        dict(zip(["question", "answer", "task"], [*create_task_fn(category=category), task_name]))
        for category in sorted(set(p.category for p in data))
    ]
    return Dataset.from_list(tasks)


# Rewards


def reward_exact_numeric_match(completion: vf.Messages, parser: vf.Parser, answer: str, **_) -> float:
    guess: str = parser.parse_answer(completion) or ""
    return 1.0 if guess == answer else 0.0


# Environment


def load_environment(
    think: bool = True,
    allow_tool_use: bool = True,
    num_items: int = 100,
    num_categories: int = 10,
    locale: str = "en",
    seed: int = 42,
    task_configs: dict[str, dict] = {
        SumPriceByCategory.name: {},
        TopKExpensiveByCategorySorted.name: {"top_k": 3},
    },
    **kwargs,
) -> vf.Environment:
    csv_data, data = create_catalog(
        n=num_items,
        n_categories=num_categories,
        seed=seed,
        locale=locale,
    )
    tasks: list[Task] = [
        t(**task_configs[t.name])
        for t in (
            SumPriceByCategory,
            TopKExpensiveByCategorySorted,
        )
        if t.name in task_configs
    ]
    datasets: list[Dataset] = [task.build_dataset(data, csv_data) for task in tasks]
    think_prompt = (
        "You are a CSV insight and arithmetic expert. Think step-by-step inside <think>...</think> tags, then answer following the specified format.\n"
        if think
        else ""
    )
    tool_use_prompt = "\nYou can use the provided tools to help answer the question." if allow_tool_use else ""
    return vf.EnvGroup(
        [
            vf.ToolEnv(
                name=task.name,
                dataset=dataset,
                system_prompt=think_prompt + task.prompt + tool_use_prompt,
                parser=task.parser,
                rubric=task.rubric_group(),
                tools=[csv_filter, csv_agg] if allow_tool_use else [],
                max_turns=3 if allow_tool_use else 1,  # tool use needs at least 2 turns (use tool + answer)
                **kwargs[task.name] if task.name in kwargs else {},
            )
            for task, dataset in zip(tasks, datasets)
        ],
        env_names=[t.name for t in tasks],
    )
