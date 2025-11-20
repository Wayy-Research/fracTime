"""
Model registry for cataloging available forecasting models.

Provides a centralized registry of all available models with their
characteristics, requirements, and default configurations.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """
    Information about a registered model.

    Attributes:
        name: Model name
        model_class: The model class
        description: Brief description
        category: 'fractal', 'baseline', 'bayesian', 'ensemble'
        requires: List of required packages
        default_params: Default parameters for initialization
        characteristics: Model characteristics (e.g., 'handles_seasonality', 'probabilistic')
    """
    name: str
    model_class: type
    description: str
    category: str
    requires: List[str]
    default_params: Dict[str, Any]
    characteristics: List[str]


class ModelRegistry:
    """
    Registry of all available forecasting models.

    Maintains a catalog of models with their metadata, enabling:
    - Discovery of available models
    - Automatic instantiation with sensible defaults
    - Filtering by characteristics or requirements

    Example:
        >>> from fractime.selection import ModelRegistry
        >>>
        >>> registry = ModelRegistry()
        >>> registry.discover_models()
        >>>
        >>> # Get all available models
        >>> models = registry.list_models()
        >>>
        >>> # Get models by category
        >>> fractal_models = registry.get_by_category('fractal')
        >>>
        >>> # Instantiate a model
        >>> model = registry.create_model('Fractal')
    """

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._auto_discovered = False

    def register(
        self,
        name: str,
        model_class: type,
        description: str,
        category: str,
        requires: Optional[List[str]] = None,
        default_params: Optional[Dict[str, Any]] = None,
        characteristics: Optional[List[str]] = None
    ) -> None:
        """
        Register a model in the registry.

        Args:
            name: Unique model name
            model_class: The model class
            description: Brief description
            category: Model category
            requires: Required packages (for optional models)
            default_params: Default initialization parameters
            characteristics: Model characteristics
        """
        self._models[name] = ModelInfo(
            name=name,
            model_class=model_class,
            description=description,
            category=category,
            requires=requires or [],
            default_params=default_params or {},
            characteristics=characteristics or []
        )

    def discover_models(self) -> None:
        """
        Auto-discover all available models.

        Attempts to import and register:
        - Fractal models
        - Baseline models (if installed)
        - Bayesian models (if installed)
        """
        if self._auto_discovered:
            return

        # Register core fractal models
        self._register_fractal_models()

        # Register baseline models (optional)
        self._register_baseline_models()

        # Register Bayesian models (optional)
        self._register_bayesian_models()

        self._auto_discovered = True

    def _register_fractal_models(self) -> None:
        """Register fractal forecasting models."""
        try:
            import fractime as ft

            self.register(
                name='Fractal',
                model_class=ft.FractalForecaster,
                description='Classical fractal forecasting with Hurst exponent',
                category='fractal',
                requires=[],
                default_params={},
                characteristics=['probabilistic', 'handles_long_memory', 'fast']
            )
        except ImportError:
            pass

    def _register_baseline_models(self) -> None:
        """Register baseline models (ARIMA, GARCH, Prophet)."""
        # ARIMA
        try:
            from fractime.baselines import ARIMAForecaster

            self.register(
                name='ARIMA',
                model_class=ARIMAForecaster,
                description='Auto-regressive Integrated Moving Average',
                category='baseline',
                requires=['pmdarima'],
                default_params={'max_p': 5, 'max_q': 5, 'stepwise': True},
                characteristics=['handles_autocorrelation', 'fast']
            )
        except ImportError:
            pass

        # GARCH
        try:
            from fractime.baselines import GARCHForecaster

            self.register(
                name='GARCH',
                model_class=GARCHForecaster,
                description='Generalized Autoregressive Conditional Heteroskedasticity',
                category='baseline',
                requires=['arch'],
                default_params={'p': 1, 'q': 1},
                characteristics=['handles_volatility_clustering', 'probabilistic']
            )
        except ImportError:
            pass

        # Prophet
        try:
            from fractime.baselines import ProphetForecaster

            self.register(
                name='Prophet',
                model_class=ProphetForecaster,
                description='Facebook Prophet forecasting',
                category='baseline',
                requires=['prophet'],
                default_params={},
                characteristics=['handles_seasonality', 'handles_trends', 'probabilistic']
            )
        except ImportError:
            pass

    def _register_bayesian_models(self) -> None:
        """Register Bayesian models."""
        try:
            from fractime import BayesianFractalForecaster

            if BayesianFractalForecaster is not None:
                # Fast mode
                self.register(
                    name='Bayesian (Fast)',
                    model_class=BayesianFractalForecaster,
                    description='Bayesian fractal with ADVI (fast)',
                    category='bayesian',
                    requires=['pymc', 'arviz'],
                    default_params={'mode': 'fast', 'n_samples': 1000},
                    characteristics=['probabilistic', 'uncertainty_quantification', 'fast']
                )

                # Hybrid mode (RECOMMENDED)
                self.register(
                    name='Bayesian (Hybrid)',
                    model_class=BayesianFractalForecaster,
                    description='Bayesian fractal with hybrid sampling (recommended)',
                    category='bayesian',
                    requires=['pymc', 'arviz'],
                    default_params={'mode': 'hybrid', 'n_samples': 1000},
                    characteristics=['probabilistic', 'uncertainty_quantification', 'recommended']
                )
        except ImportError:
            pass

    def list_models(self, available_only: bool = True) -> List[str]:
        """
        List all registered models.

        Args:
            available_only: Only list models whose requirements are satisfied

        Returns:
            List of model names
        """
        if not self._auto_discovered:
            self.discover_models()

        if not available_only:
            return list(self._models.keys())

        # Check which models are available
        available = []
        for name, info in self._models.items():
            if self._check_requirements(info.requires):
                available.append(name)

        return available

    def get_by_category(self, category: str) -> List[str]:
        """Get models by category."""
        if not self._auto_discovered:
            self.discover_models()

        return [
            name for name, info in self._models.items()
            if info.category == category and self._check_requirements(info.requires)
        ]

    def get_by_characteristics(self, characteristics: List[str]) -> List[str]:
        """Get models with specific characteristics."""
        if not self._auto_discovered:
            self.discover_models()

        matching = []
        for name, info in self._models.items():
            if all(char in info.characteristics for char in characteristics):
                if self._check_requirements(info.requires):
                    matching.append(name)

        return matching

    def create_model(self, name: str, **kwargs) -> Any:
        """
        Create an instance of a registered model.

        Args:
            name: Model name
            **kwargs: Override default parameters

        Returns:
            Instantiated model

        Raises:
            ValueError: If model not found or requirements not met
        """
        if not self._auto_discovered:
            self.discover_models()

        if name not in self._models:
            raise ValueError(f"Model '{name}' not found. Available: {self.list_models()}")

        info = self._models[name]

        # Check requirements
        if not self._check_requirements(info.requires):
            missing = [req for req in info.requires if not self._is_package_available(req)]
            raise ValueError(f"Model '{name}' requires: {missing}")

        # Merge default params with overrides
        params = {**info.default_params, **kwargs}

        # Instantiate
        return info.model_class(**params)

    def get_info(self, name: str) -> ModelInfo:
        """Get information about a model."""
        if not self._auto_discovered:
            self.discover_models()

        if name not in self._models:
            raise ValueError(f"Model '{name}' not found")

        return self._models[name]

    def _check_requirements(self, requires: List[str]) -> bool:
        """Check if all requirements are satisfied."""
        return all(self._is_package_available(pkg) for pkg in requires)

    def _is_package_available(self, package_name: str) -> bool:
        """Check if a package is available."""
        try:
            __import__(package_name)
            return True
        except ImportError:
            return False

    def print_registry(self) -> None:
        """Print a formatted registry of all models."""
        if not self._auto_discovered:
            self.discover_models()

        print("=" * 80)
        print("MODEL REGISTRY")
        print("=" * 80)

        by_category = {}
        for name, info in self._models.items():
            category = info.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((name, info))

        for category in sorted(by_category.keys()):
            print(f"\n{category.upper()} MODELS:")
            print("-" * 80)

            for name, info in by_category[category]:
                available = self._check_requirements(info.requires)
                status = "✓" if available else "✗"

                print(f"\n{status} {name}")
                print(f"   {info.description}")
                if info.characteristics:
                    print(f"   Characteristics: {', '.join(info.characteristics)}")
                if info.requires:
                    print(f"   Requires: {', '.join(info.requires)}")

        print("\n" + "=" * 80)


# Global registry instance
_global_registry = None


def get_global_registry() -> ModelRegistry:
    """Get the global model registry (singleton)."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelRegistry()
        _global_registry.discover_models()
    return _global_registry


def register_model(
    name: str,
    model_class: type,
    description: str,
    category: str = 'custom',
    **kwargs
) -> None:
    """
    Register a custom model in the global registry.

    Args:
        name: Model name
        model_class: The model class
        description: Brief description
        category: Model category (default 'custom')
        **kwargs: Additional ModelInfo fields
    """
    registry = get_global_registry()
    registry.register(name, model_class, description, category, **kwargs)
