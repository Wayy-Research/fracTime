# Changelog

All notable changes to fracTime will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2026-02-06

### Added
- Cloud-Conditioned LSTM baseline forecaster (experimental)
  - Novel architecture: LSTM + FracTime simulation cloud via FiLM conditioning
  - Cloud encoder converts simulation paths to statistical features
  - MC dropout uncertainty quantification
  - Optional PyTorch dependency (gracefully skipped if not installed)
- Evaluation notebook for cloud LSTM vs other baselines
- Research study runner script

### Fixed
- CI: Made PyTorch-dependent code conditional to prevent import failures
- CI: Fixed publish workflow to require test pass before publishing
- CI: Removed stale test files referencing removed modules (fractime.core, fractime.backtesting, FractalForecaster)
- Added cloud_weight parameter validation in CloudConditionedLSTMForecaster

## [0.5.0] - 2026-02-05

### Added
- HMM-based regime detection (RegimeDetector)
- Regime-based trading strategy framework (RegimeStrategy)
- FractalLSTM hybrid architecture baseline
- Transaction Cost Analysis (TCA) module
- Multifractal DFA (MF-DFA) analysis
- Research tools: metrics, statistical tests, experiment tracking

## [0.4.0] - 2026-02-04

### Changed
- Replaced plotly with wrchart for all visualizations

## [0.3.0] - 2026-02-03

### Changed
- Complete API refactor: simplified public interface
- Renamed FractalForecaster to Forecaster
- Renamed FractalAnalyzer to Analyzer
