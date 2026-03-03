# Investigation Notes: Kwargs Validation

Looking into issue #142 - checking if functions properly reject invalid kwargs.

## Background

Issue #142 is about making sure functions raise TypeError for invalid keyword arguments instead of silently ignoring them. PR #143 already fixed this for hdi(), eti(), kde(), and histogram().

Wanted to see if other functions validate kwargs too.

## What I did

Wrote some test scripts to check the main functions. My approach:
- Tested 24+ functions with invalid kwargs like `invalid_kwarg="test"`
- Also tried common typos (`dims` vs `dim`, `methods` vs `method`) 
- Checked both direct calls and `.azstats` accessor methods

## Results

Good news - everything works! All the functions I tested properly reject invalid kwargs.

Functions tested:
- Sampling diagnostics: ess(), rhat(), mcse(), bfmi(), diagnose()
- Summary stats: summary(), mean(), median(), mode(), ci_in_rope()
- Density/viz: hdi(), eti(), kde(), histogram(), qds(), ecdf()
- LOO stuff: loo(), compare()
- Metrics: bayesian_r2(), metrics()
- Other: thin(), weight_predictions(), bayes_factor(), psense()

### How it works

Functions with `**kwargs` either pass them downstream (where they get validated) or check for unexpected kwargs explicitly and raise TypeError.

Two error patterns I saw:
- `"got an unexpected keyword argument..."` (most common)
- `"got multiple values for argument 'dims'"` (when typo conflicts with internal param)

Both work fine for catching bad inputs.

## The actual problem

Only 4 functions have tests for kwargs validation (from PR #143). The functionality is there, but test coverage is incomplete.

So I added 26 tests covering all the functions above. Tests check both direct calls and accessor methods, plus common typo scenarios.

## Test results

```
tests/test_kwargs_validation.py: 26 passed
Full test suite: 2475 passed, 2 skipped
```

## Tests added

`tests/test_kwargs_validation.py` has 26 tests organized into:
- TestKwargsValidationSamplingDiagnostics (5 tests)
- TestKwargsValidationSummaryStatistics (5 tests)
- TestKwargsValidationVisualizationFunctions (2 tests)
- TestKwargsValidationLOOFunctions (2 tests)
- TestKwargsValidationMetrics (2 tests)
- TestKwargsValidationOtherFunctions (4 tests)
- TestKwargsValidationAccessors (6 tests)

## Why add tests if validation already works?

- Prevents regressions if someone refactors the validation logic later
- Documents expected behavior for new contributors
- Covers edge cases and common typos
- Matches the test coverage level I saw elsewhere in the codebase

Issue #142 is technically resolved functionality-wise, but this adds the test documentation we were missing.
