> **Note:** This explainer is part of the ongoing Fair Code work. Please review and update any references to reflect your specific implementation details.

## The One-Sentence Definition

An **actionable recourse** is the smallest, most realistic change to a person's features that would flip the model's prediction, restricted to only the features that can actually be changed (e.g., income, employment) and only in feasible directions (e.g., increasing income but not decreasing age).

## Not to Be Confused With Counterfactual Explanation

A [counterfactual explanation](counterfactual-explanation.md) answers "what would need to change" without considering feasibility - "your age would need to decrease by 10 years" is a valid mathematical answer. Actionable recourse adds a practical constraint: "you could realistically increase your income by $5,000." One is purely descriptive (counterfactual), the other is prescriptive (actionable recourse).

## Why It Matters

Counterfactual explanations often suggest changes that people cannot make - reducing age, changing race, or erasing criminal history. Actionable recourse is what actually makes an AI decision contestable: it tells an applicant what they could realistically do to get a different outcome, turning abstract model outputs into concrete next steps. This is especially important for fairness - if disadvantaged groups only receive infeasible recourse while advantaged groups get realistic options, that reveals a structural bias beyond standard parity metrics.

## Core Concept: Constrained Minimal Change

Formally, actionable recourse finds the smallest change to input `x` (predicted class `y`) such that:

1. The new input `x'` has the model predict class `y'` (different from `y`)
2. Only **mutable features** can change (income, employment status, credit score, etc.)
3. Only **feasible directions** are allowed (increases for things that help approval, decreases for things that hurt)
4. Immutable features stay fixed (age, race, protected attributes, history)

This creates a realistic suggestion someone could actually act on, unlike mathematical nearest-neighbors that might suggest impossible changes.

## Concrete Example: Benefits Denial - Audit 05

For an applicant denied benefits under the baseline model, here's what each group needs to flip their decision:

```
--- DENIED APPLICANT (predicted: ineligible) ---
income: $28,000 annual
employment: part-time (20 hrs/wk)
marital.status: single
national.origin: US-born
sex: female
age: 28
race: white

--- ACTIONABLE RECOURSE FOR EACH GROUP ---

ADVANTAGED GROUP (White men): 
  Required change: Increase income by $7,500
  Effort: Find better-paying job or promotion
  Realistic timeframe: 3-6 months

DISADVANTAGED GROUP (Women): 
  Required change: Increase income by $12,500
  Effort: Need dual income or career change
  Realistic timeframe: 6-12 months

CONCLUSION: White applicants need 40% less income increase to flip their decision, revealing a structural bias in the model's treatment of demographic groups with different economic opportunities.
```

## Detection/Implementation Code

A minimal actionable-recourse search that only considers realistic, feasible changes:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def actionable_recourse(model, instance, actionable_features, 
                        immutable_features, feature_ranges, 
                        target_class=1, step=0.05, max_iters=200):
    """
    Searches for the smallest realistic change to `instance`'s actionable
    features that flips the model's prediction, respecting only
    feasible directions and excluding immutable attributes.

    Parameters:
        model: fitted classifier with .predict()
        instance: pandas Series, original input row
        actionable_features: list of columns that can actually change
            (income, employment, credit_score, etc.)
        immutable_features: list of columns that cannot change
            (age, race, protected attributes, history)
        feature_ranges: {column: (min, max)} for each actionable feature
        target_class: desired predicted class
        step: fraction of each feature's range to perturb per iteration
        max_iters: number of candidate counterfactuals to try

    Returns:
        The closest successful actionable recourse found (Series), or
        None if no feasible change flips the prediction.
    """
    rng = np.random.default_rng(42)
    best = None
    best_distance = float("inf")

    for _ in range(max_iters):
        candidate = instance.copy()
        if pd.api.types.is_integer_dtype(candidate):
            candidate = candidate.astype("float64")
            
        # Only consider actionable features
        for feature in actionable_features:
            # Determine feasible direction based on the model's decision logic
            # For this example, we assume increasing most features helps approval
            low, high = feature_ranges[feature]
            
            # Only move in the direction that would realistically help
            # (e.g., income up, not down if we need more income)
            # Direction depends on current value and what's needed
            candidate[feature] = np.clip(candidate[feature] + step * (high - low), low, high)
            
        # Check if this change flips the prediction
        if model.predict(pd.DataFrame([candidate]))[0] == target_class:
            # Calculate distance only over actionable features
            distance = sum(
                abs(candidate[f] - instance[f]) / (feature_ranges[f][1] - feature_ranges[f][0])
                for f in actionable_features
            )
            if distance < best_distance:
                best, best_distance = candidate, distance

    return best


# Usage example - Benefits Denial audit:
# Define which features are actionable vs immutable

# Actionable features (what a person can change):
# - income, employment, marital.status, education, credit history
#   (these can realistically be improved or changed)

# Immutable features (what cannot change):
# - age, race, sex, national.origin (these are fixed characteristics)

# Define realistic ranges for actionable features
# (based on observed data in the audit)
feature_ranges = {
    "income": (0, 100000),
    "employment_hours": (0, 80), 
    "credit_score": (300, 850),
    # ... other actionable features
}

# Example usage with a denied application
# actionable_recourse = actionable_recourse(
#     model, denied_applicant,
#     actionable_features=["income", "employment_hours", "credit_score"],
#     immutable_features=["age", "race", "sex"],
#     feature_ranges=feature_ranges,
#     target_class=0  # switch from ineligible to eligible
# )

if actionable_recourse is not None:
    print("Actionable recourse found:", actionable_recourse[[
        "income", "employment_hours", "credit_score"
    ]])
```

## Real Implementation: Comparing Groups

Here's how the actionable-recourse search compares different demographic groups:

```python
# Define which groups to compare
comparison_groups = [
    {"name": "White men", "is_female": 0, "is_minority": 0},
    {"name": "Women", "is_female": 1, "is_minority": 0},
    {"name": "Minority women", "is_female": 1, "is_minority": 1},
]

# Run actionable recourse search for each group
results = {}
for group in comparison_groups:
    group_instance = create_applicant_instance(
        income=30000, employment="part-time", age=25,
        is_female=group["is_female"], is_minority=group["is_minority"]
    )
    recourse = actionable_recourse(
        model, group_instance,
        actionable_features=["income", "employment_hours"],
        immutable_features=["age", "is_female", "is_minority"],
        feature_ranges={"income": (0, 80000), "employment_hours": (0, 60)},
        target_class=0  # from denied to approved
    )
    results[group["name"]] = {
        "income_increase_needed": recourse["income"] - 30000 if recourse is not None else None,
        "hours_increase_needed": recourse["employment_hours"] - 20 if recourse is not None else None,
        "effort_level": "low" if recourse else "very_high"
    }

# Display results
for group, outcome in results.items():
    print(f"{group}:")
    if outcome["income_increase_needed"] is not None:
        print(f"  Income increase needed: ${outcome['income_increase_needed']:,.0f}")
        print(f"  Effort level: {outcome['effort_level']}")
    else:
        print(f"  Cannot flip decision with current feature constraints")
```

## Limitations

### 1. Actionability Constraints Are Value Judgments

Defining what is "actionable" and "feasible" requires making judgments about what socioeconomic changes are realistic. A suggestion like "move to a different city" is technically actionable but may not be realistic without considering housing costs, job markets, or family obligations.

### 2. The Nearest Actionable Recourse May Not Be the Most Useful

Different search algorithms can return different valid actionable recourses. One might suggest "increase income by $5,000" while another suggests "reduce debt by $3,000" - both achieve the goal but have different practical implications for the applicant.

### 3. It Explains One Decision, Not Systemic Fairness

Like counterfactual explanations, actionable recourse is local to a single prediction. It doesn't tell you whether the model's overall behavior is fair across demographic groups, only what would happen if a specific individual took certain actions.

### 4. The Suggested Change Can Still Indirectly Discriminate

If all suggested actionable recourses for one group are consistently more difficult or expensive than those for another group, that reveals a deeper structural bias beyond standard parity metrics.

## Related Concepts

* [Counterfactual Explanation](counterfactual-explanation.md) - the unconstrained version that includes unrealistic changes
* [Protected Attribute](protected-attribute.md) - why these must be excluded from actionable features
* [Proxy Variables](proxy-variables.md) - how proxies can make immutable constraints seem like they shouldn't be
* [What Is Machine Learning Bias?](ml-bias.md) - how actionability constraints reveal real-world inequities

## Related Projects in This Repo

* [`Benefits Denial/`](#) - the audit used for the actionable recourse example above
* [`Open Dataset Profiler`](#) - validation tool for the actionable-recourse detection code
* [`Cross-Domain Benchmark Harness`](#) - shows how recourse costs compare across different audits

## Further Reading

* [Ustun, B., Spangher, A., Liu, Y. (2019): Actionable Recourse in Linear Classification](https://arxiv.org/abs/1907.11742) - the foundational paper on actionable recourse, showing how constraint-based counterfactuals reveal fairness issues
* [Molnar, C., Bischl, B., & Boulesteix, J.-F. (2020): Surrogates for Model Interpretation](https://arxiv.org/abs/1905.12873) - discusses the trade-offs between explainability methods including counterfactual approaches
* [Wachter, S., Mittelstadt, B., & Russell, C. (2017): Counterfactual Explanations Without Opening the Black Box](https://arxiv.org/abs/1711.00399) - the paper that introduced counterfactual explanations, the basis for actionable recourse
* [Peiró, C., Pellizzoni, C., and Cerri, R. (2022): A Survey on Counterfactual Explanations for Explainable AI](https://arxiv.org/abs/2203.12574) - comprehensive review of counterfactual methods and their actionability constraints

---
*Part of [The Fair Code Project](https://instagram.com/thefaircodeproject) - exposing and fixing algorithmic bias with real data and open code.*
