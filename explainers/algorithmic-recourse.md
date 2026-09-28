> _A model can tell someone what to change. Recourse asks whether that change is possible for them, and whether other people face a harder path to the same decision._

## The One-Sentence Definition

**Algorithmic recourse** is a feasible, actionable change to a person's circumstances or inputs that would change an unfavorable model decision into a favorable one.

## Why It Matters

A **counterfactual explanation** describes what input change would flip a prediction. Recourse adds the person's real constraints: can they make that change, in the time available, at a cost they can bear? A model might say that a rejected applicant would be hired with a higher test score. That is a description of the model's boundary; it is not yet a useful recommendation if the applicant cannot access preparation or a retest.

That distinction matters for fairness. Two people can receive the same rejection and the same nominal advice while facing very different costs to act on it. Comparing recourse across groups asks whether people have similarly reachable paths to a favorable outcome. This is different from [Demographic Parity](demographic-parity.md) or [Disparate Impact](disparate-impact.md), which compare group outcome rates, not the effort or feasibility of changing an individual decision. Unequal recourse can expose a burden that an outcome-rate metric does not measure, but it does not by itself prove why the burden differs or establish discrimination.

## Core Concept: Constrained Search and Actionability

For a denied person, search for a changed input that the model accepts while keeping immutable features fixed. Among successful candidates, minimize a stated cost, such as the sum of each allowed change divided by that feature's observed training range. The search is constrained twice: only designated actionable features may change, and each feature may move only in an allowed direction and within a stated bound.

Mutability is a domain decision, not a property a model can discover. Depending on the decision, potentially mutable features might include income, employment tenure, a credit score over time, debt, requested loan amount, or address. Each has conditions: income depends on access to jobs, credit repair takes time and money, and moving may be impossible. Immutable examples include age, race, gender, birthplace, and completed past events or history. Past experience cannot be rewritten, although a person's accumulated years of experience can increase over time.

The example below uses a hiring audit. It permits only `Experience_Years` and `Technical_Test_Score` to increase, up to their maxima in the training split. Gender and age remain unchanged, as do all other fields. That direction rule is explicit, but it is only a simple actionability assumption: it does not claim that more experience or a higher test score can be obtained immediately or fairly.

Ustun, Spangher, and Liu's _Actionable Recourse in Linear Classification_ (ACM FAT\* 2019, now ACM FAccT) formalizes recourse for linear classifiers and presents integer-programming methods for finding actionable changes. The paper shows that standard modeling choices can materially affect recourse and motivates measuring it; it does not establish that this example's group difference is causal or generalizable.

## Concrete Example: AI Fair Recruitment - Audit 02

This example uses the real `AI Fair Recruitment/AI_Fair_Recruitment_Dataset.csv` and reproduces the audit's `unfair.py` baseline: 121,190 complete rows, an 80/20 split with `random_state=42`, and a 100-tree `RandomForestClassifier` with `random_state=42`. The audit's four input fields become five model columns after one-hot encoding Gender: Gender, Age, Experience_Years, and Technical_Test_Score. The decision is the dataset's binary `Hiring_Decision`, predicted at the classifier's default threshold. The audit manifest identifies Female as the disadvantaged group and Male as the advantaged group; `Other` is not included in this two-group comparison.

The checked-in **current** aggregate benchmark at `results/results_fairness.csv` reports a baseline random-forest demographic-parity difference (Female minus Male) of **-4.77 percentage points**, with a 95% interval of **[-5.86, -3.69] points** and a permutation p-value displayed as `0.0000` (rounded). That is an outcome-rate result, not a recourse-cost result.

For the separate recourse calculation below, the code takes a reproducible random sample of 100 denied Female applicants and 100 denied Male applicants from the fixed test split. For each, it exhaustively tests integer increases in experience and test score, holding Gender, Age, and every other model column fixed. Cost is normalized L1 distance: the sum of each increase divided by that feature's training-set range. Candidates are checked in full against the same fitted baseline model.

| Test-set group | Sampled denied applicants | Found a path within bounds | Median normalized cost | Median experience increase | Median test-score increase |
| -------------- | ------------------------: | -------------------------: | ---------------------: | -------------------------: | -------------------------: |
| Female         |                       100 |                    100/100 |                0.06672 |                    0 years |                   4 points |
| Male           |                       100 |                     99/100 |                0.06061 |                    0 years |                   4 points |

In this sample, the Female median cost is about **10.1% higher** than the Male median among applicants with a successful search. This is a descriptive result for two seeded samples, not a significance-tested or population-wide fairness finding. One sampled Male applicant had no successful candidate within the allowed features and observed bounds. The similar median changes do not mean the underlying effort is equal: the normalized cost combines two dimensions and excludes time, money, access, and the effort of improving a test score.

## Detection Code

Run from the repository root with `python3` and the project's dependencies installed. The script fits the audited baseline and prints both groups' sample success rates and median costs. It raises an error for missing features, empty comparison groups, or unusable feature ranges; applicants with no found path are retained in the success-rate denominator and excluded only from the median successful cost.

```python
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def nearest_recourse(
    model: RandomForestClassifier,
    denied_row: pd.Series,
    actionable_features: List[str],
    upper_bounds: Dict[str, int],
    feature_ranges: Dict[str, float],
) -> Optional[Tuple[pd.Series, float]]:
    """Find the minimum-cost successful candidate using increases only."""
    missing = set(actionable_features) - set(denied_row.index)
    if missing:
        raise ValueError(f"Actionable features are missing from the row: {sorted(missing)}")

    for feature in actionable_features:
        if feature not in upper_bounds or feature not in feature_ranges:
            raise ValueError(f"Missing bound or range for {feature!r}")
        if feature_ranges[feature] <= 0:
            raise ValueError(f"Feature range must be positive for {feature!r}")
        if denied_row[feature] > upper_bounds[feature]:
            return None

    value_grid = list(product(*(
        range(int(denied_row[feature]), upper_bounds[feature] + 1)
        for feature in actionable_features
    )))
    candidates = pd.DataFrame(
        [denied_row.to_dict() for _ in value_grid], columns=denied_row.index
    )
    for feature_index, feature in enumerate(actionable_features):
        candidates[feature] = [values[feature_index] for values in value_grid]

    costs = [
        sum(
            (values[index] - denied_row[feature]) / feature_ranges[feature]
            for index, feature in enumerate(actionable_features)
        )
        for values in value_grid
    ]
    successful = np.flatnonzero(model.predict(candidates) == 1)
    if not successful.size:
        return None

    best_index = min(successful, key=lambda index: costs[index])
    return candidates.iloc[best_index].copy(), float(costs[best_index])


def compare_group_recourse(
    model: RandomForestClassifier,
    test_features: pd.DataFrame,
    group_labels: pd.Series,
    training_features: pd.DataFrame,
    actionable_features: List[str],
    groups: List[str],
    sample_size: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure constrained recourse for a seeded sample of denied cases."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not test_features.index.equals(group_labels.index):
        raise ValueError("Group labels must be aligned with test_features")

    upper_bounds = {
        feature: int(training_features[feature].max())
        for feature in actionable_features
    }
    feature_ranges = {
        feature: float(training_features[feature].max() - training_features[feature].min())
        for feature in actionable_features
    }
    predictions = model.predict(test_features)
    rng = np.random.default_rng(random_state)
    records = []

    for group in groups:
        denied_ids = test_features.index[
            (group_labels == group).to_numpy() & (predictions == 0)
        ].to_numpy()
        if not denied_ids.size:
            raise ValueError(f"No denied test cases found for group {group!r}")
        sampled_ids = rng.choice(
            denied_ids, size=min(sample_size, len(denied_ids)), replace=False
        )

        for row_id in sampled_ids:
            original = test_features.loc[row_id]
            result = nearest_recourse(
                model, original, actionable_features, upper_bounds, feature_ranges
            )
            if result is None:
                records.append({"group": group, "found": False, "cost": np.nan,
                                "experience_increase": np.nan,
                                "score_increase": np.nan})
                continue

            changed, cost = result
            records.append({
                "group": group,
                "found": True,
                "cost": cost,
                "experience_increase": (
                    changed["Experience_Years"] - original["Experience_Years"]
                ),
                "score_increase": (
                    changed["Technical_Test_Score"] - original["Technical_Test_Score"]
                ),
            })

    return pd.DataFrame(records)


def main() -> None:
    """Fit the audit baseline and print sampled recourse by gender."""
    dataset_path = (
        Path.cwd() / "AI Fair Recruitment" / "AI_Fair_Recruitment_Dataset.csv"
    )
    if not dataset_path.is_file():
        raise FileNotFoundError("Run this script from the Fair-Code repository root")

    data = pd.read_csv(dataset_path)
    required = [
        "Hiring_Decision", "Gender", "Age",
        "Experience_Years", "Technical_Test_Score",
    ]
    data = data.dropna(subset=required)
    model_features = [
        "Gender", "Age", "Experience_Years", "Technical_Test_Score"
    ]
    actionable_features = ["Experience_Years", "Technical_Test_Score"]
    features = pd.get_dummies(data[model_features], drop_first=True)
    labels = data["Hiring_Decision"]
    train_x, test_x, _, _ = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train_x, labels.loc[train_x.index])
    groups = data.loc[test_x.index, "Gender"]
    results = compare_group_recourse(
        model=model,
        test_features=test_x,
        group_labels=groups,
        training_features=train_x,
        actionable_features=actionable_features,
        groups=["Female", "Male"],
        sample_size=100,
        random_state=42,
    )

    for group, sample in results.groupby("group", sort=False):
        successful = sample[sample["found"]]
        median_cost = successful["cost"].median()
        median_experience = successful["experience_increase"].median()
        median_score = successful["score_increase"].median()
        print(
            f"{group}: {len(sample)} sampled; {len(successful)} found; "
            f"median cost={median_cost:.5f}; "
            f"median experience increase={median_experience:.0f}; "
            f"median score increase={median_score:.0f}"
        )


if __name__ == "__main__":
    main()
```

## Limitations

### Actionability is a value judgment

Developers, domain experts, policy-makers, and affected communities should decide which changes count as actionable and what time and cost bounds are acceptable. The code's one-way increases and training-set maxima are modeling choices, not facts about what an applicant can do.

### A direction constraint is not a real-world plan

More employment experience takes time and depends on job access. A higher test score may require money, preparation, accommodations, or another chance to take the test. Credit repair, increased income, and relocation have similar constraints. A mathematically valid path can still be economically infeasible or unavailable to a particular group.

### The audit cannot measure all barriers

The hiring dataset does not measure access to training, local job availability, discrimination in hiring, caregiving responsibilities, disability accommodations, or the cost of improving a score. The two-group sample also does not report intersectional results, and its random sample is not a causal or population-wide estimate.

### Recourse does not fix the decision system

An applicant should not have to overcome a discriminatory threshold to receive fair treatment. Recourse does not guarantee fairness, validate the model's target, or replace aggregate audits and systemic changes to hiring access and practice. A useful recourse process needs review, transparency, and a way to challenge the decision itself.

## Related Concepts

- [Counterfactual Explanation](counterfactual-explanation.md) - a description of what input change flips one prediction; recourse adds feasibility and actionability.
- [Disparate Treatment](disparate-treatment.md) - direct use of a protected attribute, which the hiring audit's baseline model includes.
- [Fairness Through Unawareness](fairness-through-unawareness.md) - why removing a protected input alone does not guarantee fair outcomes.
- [Demographic Parity](demographic-parity.md) - an outcome-rate comparison, unlike the individual cost comparison here.
- [Ustun, B., Spangher, A., Liu, Y. (2019). _Actionable Recourse in Linear Classification_. Proceedings of the ACM Conference on Fairness, Accountability, and Transparency (FAT\* 2019).](https://doi.org/10.1145/3287560.3287566) - develops integer-programming methods for actionable recourse in linear classifiers and shows that modeling choices can affect recourse.
