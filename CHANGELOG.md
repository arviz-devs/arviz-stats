<a id="v0.6.0"></a>
# [v0.6.0](https://github.com/arviz-devs/arviz-stats/releases/tag/v0.6.0) - 2025-06-16

## What's Changed

## New Features
* Add loo_pit function by [@aloctavodia](https://github.com/aloctavodia) in [#84](https://github.com/arviz-devs/arviz-stats/pull/84)
* Add randomized loo_pit by [@aloctavodia](https://github.com/aloctavodia) in [#85](https://github.com/arviz-devs/arviz-stats/pull/85)
* Add Nested rhat by [@aloctavodia](https://github.com/aloctavodia) in [#87](https://github.com/arviz-devs/arviz-stats/pull/87)
* Add loo expectations and metrics by [@aloctavodia](https://github.com/aloctavodia) in [#89](https://github.com/arviz-devs/arviz-stats/pull/89)
* Isotonic: add group option by [@aloctavodia](https://github.com/aloctavodia) in [#92](https://github.com/arviz-devs/arviz-stats/pull/92)
* Add group argument to difference_ecdf_pit by [@aloctavodia](https://github.com/aloctavodia) in [#94](https://github.com/arviz-devs/arviz-stats/pull/94)
* Update KDE computation & Bayes factor implementation ([#55](https://github.com/arviz-devs/arviz-stats/issues/55)) by [@PiyushPanwarFST](https://github.com/PiyushPanwarFST) in [#95](https://github.com/arviz-devs/arviz-stats/pull/95)
* Adding rank normalisation methods to rhat_nested by [@zaxtax](https://github.com/zaxtax) in [#102](https://github.com/arviz-devs/arviz-stats/pull/102)
* Add loo_approximate_posterior function by [@jordandeklerk](https://github.com/jordandeklerk) in [#101](https://github.com/arviz-devs/arviz-stats/pull/101)
* Add loo_subsample function for large datasets by [@jordandeklerk](https://github.com/jordandeklerk) in [#106](https://github.com/arviz-devs/arviz-stats/pull/106)
* Add support for different variable names in loo_pit by [@jordandeklerk](https://github.com/jordandeklerk) in [#115](https://github.com/arviz-devs/arviz-stats/pull/115)
* Add R2 by [@aloctavodia](https://github.com/aloctavodia) in [#108](https://github.com/arviz-devs/arviz-stats/pull/108)
* Add Wasserstein distance and KL-divergence by [@aloctavodia](https://github.com/aloctavodia) in [#119](https://github.com/arviz-devs/arviz-stats/pull/119)
* Add "regular" metrics by [@aloctavodia](https://github.com/aloctavodia) in [#121](https://github.com/arviz-devs/arviz-stats/pull/121)
* Fix loo pit randomization by [@aloctavodia](https://github.com/aloctavodia) in [#120](https://github.com/arviz-devs/arviz-stats/pull/120)
* Add ci_in_rope by [@aloctavodia](https://github.com/aloctavodia) in [#123](https://github.com/arviz-devs/arviz-stats/pull/123)
* Add loo_moment_match by [@jordandeklerk](https://github.com/jordandeklerk) in [#126](https://github.com/arviz-devs/arviz-stats/pull/126)
* Add function-specific Pareto k-hat for loo_expectations by [@jordandeklerk](https://github.com/jordandeklerk) in [#132](https://github.com/arviz-devs/arviz-stats/pull/132)



## Maintenance and bug fixes
* Add more tests and examples for loo_metrics by [@aloctavodia](https://github.com/aloctavodia) in [#90](https://github.com/arviz-devs/arviz-stats/pull/90)
* Fix specific test failure by [@amaloney](https://github.com/amaloney) in [#99](https://github.com/arviz-devs/arviz-stats/pull/99)
* Autocorr, do not add coords by [@aloctavodia](https://github.com/aloctavodia) in [#97](https://github.com/arviz-devs/arviz-stats/pull/97)
* Start ess ess-related improvements by [@OriolAbril](https://github.com/OriolAbril) in [#44](https://github.com/arviz-devs/arviz-stats/pull/44)
* Testing infrastructure and architecture documentation by [@OriolAbril](https://github.com/OriolAbril) in [#103](https://github.com/arviz-devs/arviz-stats/pull/103)
* Fix ess prob validation by [@OriolAbril](https://github.com/OriolAbril) in [#109](https://github.com/arviz-devs/arviz-stats/pull/109)
* Fix ess prob validation by [@OriolAbril](https://github.com/OriolAbril) in [#110](https://github.com/arviz-devs/arviz-stats/pull/110)
* Fix HDF error in doc build by [@OriolAbril](https://github.com/OriolAbril) in [#113](https://github.com/arviz-devs/arviz-stats/pull/113)
* Improve publish workflow by [@OriolAbril](https://github.com/OriolAbril) in [#117](https://github.com/arviz-devs/arviz-stats/pull/117)
* Improved bayes_factor for multi-variable support by [@PiyushPanwarFST](https://github.com/PiyushPanwarFST) in [#104](https://github.com/arviz-devs/arviz-stats/pull/104)
* Fix summary bugs and add tests by [@aloctavodia](https://github.com/aloctavodia) in [#128](https://github.com/arviz-devs/arviz-stats/pull/128)
* psense: update link warning by [@aloctavodia](https://github.com/aloctavodia) in [#133](https://github.com/arviz-devs/arviz-stats/pull/133)
* enforce dim+sample_dims convention on functions by [@OriolAbril](https://github.com/OriolAbril) in [#131](https://github.com/arviz-devs/arviz-stats/pull/131)
* Remove "anes" dataset from tests by [@aloctavodia](https://github.com/aloctavodia) in [#135](https://github.com/arviz-devs/arviz-stats/pull/135)
* update tests to use arviz_base by [@aloctavodia](https://github.com/aloctavodia) in [#137](https://github.com/arviz-devs/arviz-stats/pull/137)
* Computational fixes for loo_moment_match by [@jordandeklerk](https://github.com/jordandeklerk) in [#141](https://github.com/arviz-devs/arviz-stats/pull/141)
* raise when extra kwargs present in hdi&kde, add test for eti,hdi,kde&histogram by [@OriolAbril](https://github.com/OriolAbril) in [#143](https://github.com/arviz-devs/arviz-stats/pull/143)

## Documentation
* Fix link eabm by [@aloctavodia](https://github.com/aloctavodia) in [#105](https://github.com/arviz-devs/arviz-stats/pull/105)
* Add overview guide by [@aloctavodia](https://github.com/aloctavodia) in [#129](https://github.com/arviz-devs/arviz-stats/pull/129)

## New Contributors
* [@rohanbabbar04](https://github.com/rohanbabbar04) made their first contribution in [#93](https://github.com/arviz-devs/arviz-stats/pull/93)
* [@amaloney](https://github.com/amaloney) made their first contribution in [#99](https://github.com/arviz-devs/arviz-stats/pull/99)
* [@zaxtax](https://github.com/zaxtax) made their first contribution in [#102](https://github.com/arviz-devs/arviz-stats/pull/102)
* [@jordandeklerk](https://github.com/jordandeklerk) made their first contribution in [#101](https://github.com/arviz-devs/arviz-stats/pull/101)

**Full Changelog**: https://github.com/arviz-devs/arviz-stats/compare/v0.5.0...v0.6.0

[Changes][v0.6.0]


<a id="v0.5.0"></a>
# [v0.5.0](https://github.com/arviz-devs/arviz-stats/releases/tag/v0.5.0) - 2025-03-21

## What's Changed
* Add randomized PIT by [@aloctavodia](https://github.com/aloctavodia) in [#75](https://github.com/arviz-devs/arviz-stats/pull/75)
* Simplify rank_ecdf by [@aloctavodia](https://github.com/aloctavodia) in [#76](https://github.com/arviz-devs/arviz-stats/pull/76)
* Add loo by [@aloctavodia](https://github.com/aloctavodia) in [#77](https://github.com/arviz-devs/arviz-stats/pull/77)
* Add Compare by [@aloctavodia](https://github.com/aloctavodia) in [#78](https://github.com/arviz-devs/arviz-stats/pull/78)
* Cap tail at 20% by [@aloctavodia](https://github.com/aloctavodia) in [#79](https://github.com/arviz-devs/arviz-stats/pull/79)
* Transform PIT values into ETI coverage values by [@aloctavodia](https://github.com/aloctavodia) in [#80](https://github.com/arviz-devs/arviz-stats/pull/80)


## New Contributors
* [@github-actions](https://github.com/github-actions) made their first contribution in [#73](https://github.com/arviz-devs/arviz-stats/pull/73)

**Full Changelog**: https://github.com/arviz-devs/arviz-stats/compare/v.0.4.0...v0.5.0

[Changes][v0.5.0]


<a id="v.0.4.0"></a>
# [v.0.4.0](https://github.com/arviz-devs/arviz-stats/releases/tag/v.0.4.0) - 2025-03-05

## What's Changed
* move in power_scale_dataset from arviz-plots by [@aloctavodia](https://github.com/aloctavodia) in [#38](https://github.com/arviz-devs/arviz-stats/pull/38)
* Use datatree from xarray by [@OriolAbril](https://github.com/OriolAbril) in [#35](https://github.com/arviz-devs/arviz-stats/pull/35)
* Update pyproject.toml by [@OriolAbril](https://github.com/OriolAbril) in [#42](https://github.com/arviz-devs/arviz-stats/pull/42)
* Add functions by [@aloctavodia](https://github.com/aloctavodia) in [#39](https://github.com/arviz-devs/arviz-stats/pull/39)
* Add Rhat and MCSE functions by [@aloctavodia](https://github.com/aloctavodia) in [#47](https://github.com/arviz-devs/arviz-stats/pull/47)
* fix docstring mcse by [@aloctavodia](https://github.com/aloctavodia) in [#49](https://github.com/arviz-devs/arviz-stats/pull/49)
* Add thin function by [@aloctavodia](https://github.com/aloctavodia) in [#48](https://github.com/arviz-devs/arviz-stats/pull/48)
* Add separated prior and likelihood groups by [@aloctavodia](https://github.com/aloctavodia) in [#53](https://github.com/arviz-devs/arviz-stats/pull/53)
* Priorsense functions: allow inferencedata as input by [@aloctavodia](https://github.com/aloctavodia) in [#54](https://github.com/arviz-devs/arviz-stats/pull/54)
* Add bayes_factor function by [@PiyushPanwarFST](https://github.com/PiyushPanwarFST) in [#52](https://github.com/arviz-devs/arviz-stats/pull/52)
* Add summary function by [@aloctavodia](https://github.com/aloctavodia) in [#51](https://github.com/arviz-devs/arviz-stats/pull/51)
* Add isotonic regression fit by [@aloctavodia](https://github.com/aloctavodia) in [#58](https://github.com/arviz-devs/arviz-stats/pull/58)
* isotonic_ fit, work on datatrees by [@aloctavodia](https://github.com/aloctavodia) in [#60](https://github.com/arviz-devs/arviz-stats/pull/60)
* psense: Enhance citations and references to external information sources. by [@aloctavodia](https://github.com/aloctavodia) in [#57](https://github.com/arviz-devs/arviz-stats/pull/57)
* utils functions for rootogram by [@aloctavodia](https://github.com/aloctavodia) in [#61](https://github.com/arviz-devs/arviz-stats/pull/61)
* change var_names argument to data_pairs for isotonic fit by [@aloctavodia](https://github.com/aloctavodia) in [#64](https://github.com/arviz-devs/arviz-stats/pull/64)
* Add autocorr accessor to _BaseAccessor class by [@suhaani-agarwal](https://github.com/suhaani-agarwal) in [#65](https://github.com/arviz-devs/arviz-stats/pull/65)
* Ensure we preserve number of observations by [@aloctavodia](https://github.com/aloctavodia) in [#66](https://github.com/arviz-devs/arviz-stats/pull/66)
* Add upper bound sphinx by [@aloctavodia](https://github.com/aloctavodia) in [#67](https://github.com/arviz-devs/arviz-stats/pull/67)
* Add functions for ecdf_pit and simultaneuous confidence bands computations by [@aloctavodia](https://github.com/aloctavodia) in [#69](https://github.com/arviz-devs/arviz-stats/pull/69)
* Automatic changelog and bump version by [@aloctavodia](https://github.com/aloctavodia) in [#70](https://github.com/arviz-devs/arviz-stats/pull/70)
* update dependencies for release by [@aloctavodia](https://github.com/aloctavodia) in [#72](https://github.com/arviz-devs/arviz-stats/pull/72)

## New Contributors
* [@PiyushPanwarFST](https://github.com/PiyushPanwarFST) made their first contribution in [#52](https://github.com/arviz-devs/arviz-stats/pull/52)
* [@suhaani-agarwal](https://github.com/suhaani-agarwal) made their first contribution in [#65](https://github.com/arviz-devs/arviz-stats/pull/65)

**Full Changelog**: https://github.com/arviz-devs/arviz-stats/compare/v0.3.0...v.0.4.0

[Changes][v.0.4.0]


[v0.6.0]: https://github.com/arviz-devs/arviz-stats/compare/v0.5.0...v0.6.0
[v0.5.0]: https://github.com/arviz-devs/arviz-stats/compare/v.0.4.0...v0.5.0
[v.0.4.0]: https://github.com/arviz-devs/arviz-stats/tree/v.0.4.0

<!-- Generated by https://github.com/rhysd/changelog-from-release v3.9.0 -->
