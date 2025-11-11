<a id="v0.7.0"></a>
# [v0.7.0](https://github.com/arviz-devs/arviz-stats/releases/tag/v0.7.0) - 2025-11-11

## What's Changed

### New features
* Add reloo for exact refitting in PSIS-LOO-CV by [@jordandeklerk](https://github.com/jordandeklerk) in [#148](https://github.com/arviz-devs/arviz-stats/pull/148)
* Add method for multiple chains, remove optimization method by [@aloctavodia](https://github.com/aloctavodia) in [#151](https://github.com/arviz-devs/arviz-stats/pull/151)
* Log Weights API change for LOO functions by [@jordandeklerk](https://github.com/jordandeklerk) in [#156](https://github.com/arviz-devs/arviz-stats/pull/156)
* Add warnings for finite data and fix computation for discrete data by [@aloctavodia](https://github.com/aloctavodia) in [#159](https://github.com/arviz-devs/arviz-stats/pull/159)
* Add loo_kfold for exact K-fold cross validation  by [@jordandeklerk](https://github.com/jordandeklerk) in [#158](https://github.com/arviz-devs/arviz-stats/pull/158)
* Compute isotonic regression for categorical and ordinal data by [@aloctavodia](https://github.com/aloctavodia) in [#167](https://github.com/arviz-devs/arviz-stats/pull/167)
* Organize computations for visualizations and make it easy to apply on groupbys by [@OriolAbril](https://github.com/OriolAbril) in [#162](https://github.com/arviz-devs/arviz-stats/pull/162)
* Add functionality to compute quantile dots by [@aloctavodia](https://github.com/aloctavodia) in [#171](https://github.com/arviz-devs/arviz-stats/pull/171)
* Add mode computation by [@aloctavodia](https://github.com/aloctavodia) in [#174](https://github.com/arviz-devs/arviz-stats/pull/174)
* Add support for Jacobian adjustment to elpd_loo values by [@jordandeklerk](https://github.com/jordandeklerk) in [#175](https://github.com/arviz-devs/arviz-stats/pull/175)
* Add more informative errors for model comparison in `compare()` by [@jordandeklerk](https://github.com/jordandeklerk) in [#184](https://github.com/arviz-devs/arviz-stats/pull/184)
* Add `loo_i` function for single observation PSIS-LOO-CV computation by [@jordandeklerk](https://github.com/jordandeklerk) in [#182](https://github.com/arviz-devs/arviz-stats/pull/182)
* Add correct output for `loo_kfold()` by [@jordandeklerk](https://github.com/jordandeklerk) in [#190](https://github.com/arviz-devs/arviz-stats/pull/190)
* Implement array interface ecdf by [@aloctavodia](https://github.com/aloctavodia) in [#188](https://github.com/arviz-devs/arviz-stats/pull/188)
* Add function `loo_score()` for CRPS and SCRPS using PWM identity  by [@jordandeklerk](https://github.com/jordandeklerk) in [#196](https://github.com/arviz-devs/arviz-stats/pull/196)
* Update default ci_prob value by [@aloctavodia](https://github.com/aloctavodia) in [#200](https://github.com/arviz-devs/arviz-stats/pull/200)
* Add functions to support plot_ppc_censored by [@aloctavodia](https://github.com/aloctavodia) in [#203](https://github.com/arviz-devs/arviz-stats/pull/203)
* Add subsampling support in `compare()` by [@jordandeklerk](https://github.com/jordandeklerk) in [#216](https://github.com/arviz-devs/arviz-stats/pull/216)
* Add order statistic diagnostic for `compare()` by [@jordandeklerk](https://github.com/jordandeklerk) in [#237](https://github.com/arviz-devs/arviz-stats/pull/237)



### Maintenance and bug fixes
* Remove extra metrics.py file by [@OriolAbril](https://github.com/OriolAbril) in [#152](https://github.com/arviz-devs/arviz-stats/pull/152)
* Fix function-specific k-hat computation  by [@jordandeklerk](https://github.com/jordandeklerk) in [#165](https://github.com/arviz-devs/arviz-stats/pull/165)
* Fix loo_moment_match to be more modular by [@jordandeklerk](https://github.com/jordandeklerk) in [#166](https://github.com/arviz-devs/arviz-stats/pull/166)
* Bump actions/download-artifact from 4 to 5 by [@dependabot](https://github.com/dependabot)[bot] in [#173](https://github.com/arviz-devs/arviz-stats/pull/173)
* Bump actions/checkout from 4 to 5 by [@dependabot](https://github.com/dependabot)[bot] in [#172](https://github.com/arviz-devs/arviz-stats/pull/172)
* Bump actions/setup-python from 5 to 6 by [@dependabot](https://github.com/dependabot)[bot] in [#194](https://github.com/arviz-devs/arviz-stats/pull/194)
* Fix typing issue in `compare()` by [@jordandeklerk](https://github.com/jordandeklerk) in [#198](https://github.com/arviz-devs/arviz-stats/pull/198)
* Rework PLPD and LPD approximation for `loo_subsample()` by [@jordandeklerk](https://github.com/jordandeklerk) in [#192](https://github.com/arviz-devs/arviz-stats/pull/192)
* Make hypergeom computation more robust by [@aloctavodia](https://github.com/aloctavodia) in [#199](https://github.com/arviz-devs/arviz-stats/pull/199)
* Add callable log likelihood argument to `loo_i()` by [@jordandeklerk](https://github.com/jordandeklerk) in [#201](https://github.com/arviz-devs/arviz-stats/pull/201)
* Robustify pareto_min_ss computation by [@aloctavodia](https://github.com/aloctavodia) in [#204](https://github.com/arviz-devs/arviz-stats/pull/204)
* Survival helper: use view instead of copy by [@aloctavodia](https://github.com/aloctavodia) in [#205](https://github.com/arviz-devs/arviz-stats/pull/205)
* Make pareto_khat consistent with pareto_min_ss by [@aloctavodia](https://github.com/aloctavodia) in [#206](https://github.com/arviz-devs/arviz-stats/pull/206)
* Remove unused function by [@aloctavodia](https://github.com/aloctavodia) in [#207](https://github.com/arviz-devs/arviz-stats/pull/207)
* Add Python 3.13 to tests by [@aloctavodia](https://github.com/aloctavodia) in [#209](https://github.com/arviz-devs/arviz-stats/pull/209)
* Pin numba version by [@aloctavodia](https://github.com/aloctavodia) in [#210](https://github.com/arviz-devs/arviz-stats/pull/210)
* Use a lookup table for hypergeometric computation instead of numba by [@aloctavodia](https://github.com/aloctavodia) in [#211](https://github.com/arviz-devs/arviz-stats/pull/211)
* Fix the sign of PSIS inputs in `loo_approximate_posterior()` by [@jordandeklerk](https://github.com/jordandeklerk) in [#213](https://github.com/arviz-devs/arviz-stats/pull/213)
* Add computational fixes for `loo_moment_match()` by [@jordandeklerk](https://github.com/jordandeklerk) in [#212](https://github.com/arviz-devs/arviz-stats/pull/212)
* Make thin works with arrays by [@aloctavodia](https://github.com/aloctavodia) in [#214](https://github.com/arviz-devs/arviz-stats/pull/214)
* Automatically compute reff for pareto_khat by [@aloctavodia](https://github.com/aloctavodia) in [#215](https://github.com/arviz-devs/arviz-stats/pull/215)
* Patch computations in `compare()` for subsampling by [@jordandeklerk](https://github.com/jordandeklerk) in [#217](https://github.com/arviz-devs/arviz-stats/pull/217)
* Fix stacking weights computation by [@aloctavodia](https://github.com/aloctavodia) in [#226](https://github.com/arviz-devs/arviz-stats/pull/226)
* Add support for Jacobian adjustment to subsampled PSIS-LOO-CV by [@jordandeklerk](https://github.com/jordandeklerk) in [#227](https://github.com/arviz-devs/arviz-stats/pull/227)
* Bump actions/download-artifact from 5 to 6 by [@dependabot](https://github.com/dependabot)[bot] in [#228](https://github.com/arviz-devs/arviz-stats/pull/228)
* Refactor and expand tests by [@jordandeklerk](https://github.com/jordandeklerk) in [#220](https://github.com/arviz-devs/arviz-stats/pull/220)
* Add Jacobian tests for PSIS-LOO-CV subsampling by [@jordandeklerk](https://github.com/jordandeklerk) in [#231](https://github.com/arviz-devs/arviz-stats/pull/231)
* Extend CI jobs and tox settings by [@OriolAbril](https://github.com/OriolAbril) in [#230](https://github.com/arviz-devs/arviz-stats/pull/230)
* Remove redundancy in `_split_moment_match()` by [@jordandeklerk](https://github.com/jordandeklerk) in [#235](https://github.com/arviz-devs/arviz-stats/pull/235)


### Documentation
* Improve documentation  by [@jordandeklerk](https://github.com/jordandeklerk) in [#197](https://github.com/arviz-devs/arviz-stats/pull/197)
* Improve the API formatting for `arviz_stats.compare` by [@star1327p](https://github.com/star1327p) in [#202](https://github.com/arviz-devs/arviz-stats/pull/202)
* Remove tutorial section (and duplicated link) by [@aloctavodia](https://github.com/aloctavodia) in [#208](https://github.com/arviz-devs/arviz-stats/pull/208)
* Update the "See Also" parts of API Reference by [@star1327p](https://github.com/star1327p) in [#221](https://github.com/arviz-devs/arviz-stats/pull/221)
* Add links to Testing arviz-stats page by [@star1327p](https://github.com/star1327p) in [#222](https://github.com/arviz-devs/arviz-stats/pull/222)
* Add link to EABM chapter by [@jordandeklerk](https://github.com/jordandeklerk) in [#229](https://github.com/arviz-devs/arviz-stats/pull/229)
* Fix return docstrings for loo functions by [@jordandeklerk](https://github.com/jordandeklerk) in [#233](https://github.com/arviz-devs/arviz-stats/pull/233)
* Remove extra references in `arviz_stats.rhat` by [@star1327p](https://github.com/star1327p) in [#236](https://github.com/arviz-devs/arviz-stats/pull/236)
* Fix references and bullet points in `arviz_stats.rhat_nested` by [@star1327p](https://github.com/star1327p) in [#239](https://github.com/arviz-devs/arviz-stats/pull/239)
* Add the reference Säilynoja et al. (2022) from EABM by [@star1327p](https://github.com/star1327p) in [#241](https://github.com/arviz-devs/arviz-stats/pull/241)
* Change truncation_factor name to avoid possible confusion by [@aloctavodia](https://github.com/aloctavodia) in [#240](https://github.com/arviz-devs/arviz-stats/pull/240)
* Add tests for numpy logsumexp by [@jordandeklerk](https://github.com/jordandeklerk) in [#238](https://github.com/arviz-devs/arviz-stats/pull/238)



## New Contributors
* [@star1327p](https://github.com/star1327p) made their first contribution in [#202](https://github.com/arviz-devs/arviz-stats/pull/202)

**Full Changelog**: https://github.com/arviz-devs/arviz-stats/compare/v0.6.0...v0.7.0

[Changes][v0.7.0]


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


[v0.7.0]: https://github.com/arviz-devs/arviz-stats/compare/v0.6.0...v0.7.0
[v0.6.0]: https://github.com/arviz-devs/arviz-stats/compare/v0.5.0...v0.6.0
[v0.5.0]: https://github.com/arviz-devs/arviz-stats/compare/v.0.4.0...v0.5.0
[v.0.4.0]: https://github.com/arviz-devs/arviz-stats/tree/v.0.4.0

<!-- Generated by https://github.com/rhysd/changelog-from-release v3.9.1 -->
