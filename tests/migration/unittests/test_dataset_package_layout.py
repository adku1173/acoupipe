"""Unit tests for incremental dataset package layout migration."""

# ruff: noqa: S101

from pathlib import Path

import acoupipe.datasets.synthetic as legacy_synthetic
from acoupipe.datasets import (
    Config,
    DatasetMIRACLE,
    DatasetSRIRACHA,
    DatasetSynthetic,
    DatasetSyntheticConfig,
    DatasetSyntheticISM,
    DatasetSyntheticISMConfig,
    MIRACLEConfig,
    MIRACLEParameters,
    ParameterSet,
    SRIRACHAConfig,
    SRIRACHAParameters,
    SyntheticISMConfig,
    SyntheticISMParameters,
)
from acoupipe.datasets.base import Config as BasePackageConfig
from acoupipe.datasets.base import ConfigBase as BasePackageConfigBase
from acoupipe.datasets.base import DatasetBase as BasePackageDataset
from acoupipe.datasets.base.config import Config as BaseConfigModuleConfig
from acoupipe.datasets.base.dataset import DatasetBase as BaseDatasetModuleDataset
from acoupipe.datasets.base.legacy_config import ConfigBase as BaseLegacyConfigModuleConfigBase
from acoupipe.datasets.base.legacy_config import ConfigBase as LegacyConfigBase
from acoupipe.datasets.base.parameters import ParameterSet as BaseParametersModuleParameterSet
from acoupipe.datasets.ism.config import SyntheticISMConfig as ISMSubmoduleConfig
from acoupipe.datasets.ism.dataset import DatasetSyntheticISM as ISMSubmoduleDataset
from acoupipe.datasets.ism.legacy_config import DatasetSyntheticISMConfig as ISMLegacyConfig
from acoupipe.datasets.ism.parameters import SyntheticISMParameters as ISMSubmoduleParameters
from acoupipe.datasets.miracle.config import MIRACLEConfig as MIRACLESubmoduleConfig
from acoupipe.datasets.miracle.dataset import DatasetMIRACLE as MIRACLESubmoduleDataset
from acoupipe.datasets.miracle.legacy_config import DatasetMIRACLEConfig as MIRACLELegacyConfig
from acoupipe.datasets.miracle.parameters import MIRACLEParameters as MIRACLESubmoduleParameters
from acoupipe.datasets.sriracha.config import SRIRACHAConfig as SRIRACHASubmoduleConfig
from acoupipe.datasets.sriracha.dataset import DatasetSRIRACHA as SRIRACHASubmoduleDataset
from acoupipe.datasets.sriracha.legacy_config import DatasetSRIRACHAConfig as SRIRACHALegacyConfig
from acoupipe.datasets.sriracha.parameters import SRIRACHAParameters as SRIRACHASubmoduleParameters
from acoupipe.datasets.synthetic.dataset import DatasetSynthetic as SyntheticSubmoduleDataset
from acoupipe.datasets.synthetic.legacy_config import DatasetSyntheticConfig as SyntheticLegacyConfig
from acoupipe.datasets.synthetic.legacy_config import sample_rms as synthetic_legacy_sample_rms

DATASETS_DIR = Path('src/acoupipe/datasets')


def test_datasets_top_level_contains_only_package_facade():
    """Test that dataset implementations live in packages, not top-level modules."""
    top_level_files = sorted(path.name for path in DATASETS_DIR.glob('*.py'))
    assert top_level_files == ['__init__.py']


def test_base_package_owns_dataset_and_config_classes():
    """Test that base dataset classes live in the base package."""
    assert BasePackageConfig is Config
    assert BaseConfigModuleConfig is Config
    assert Config.__module__ == 'acoupipe.datasets.base.config'
    assert BasePackageDataset is BaseDatasetModuleDataset
    assert BaseDatasetModuleDataset.__module__ == 'acoupipe.datasets.base.dataset'


def test_legacy_config_base_lives_in_base_legacy_config_module():
    """Test that ConfigBase has a base legacy module owner."""
    assert BasePackageConfigBase is LegacyConfigBase
    assert BaseLegacyConfigModuleConfigBase is LegacyConfigBase
    assert LegacyConfigBase.__module__ == 'acoupipe.datasets.base.legacy_config'


def test_parameter_set_lives_in_base_parameters_module():
    """Test that generic ParameterSet has a base package owner."""
    assert BaseParametersModuleParameterSet is ParameterSet
    assert ParameterSet.__module__ == 'acoupipe.datasets.base.parameters'


def test_non_conflicting_dataset_packages_own_parameter_classes():
    """Test that dataset ParameterSet classes live in dataset packages."""
    assert MIRACLESubmoduleParameters is MIRACLEParameters
    assert SRIRACHASubmoduleParameters is SRIRACHAParameters
    assert ISMSubmoduleParameters is SyntheticISMParameters
    assert MIRACLEParameters.__module__ == 'acoupipe.datasets.miracle.parameters'
    assert SRIRACHAParameters.__module__ == 'acoupipe.datasets.sriracha.parameters'
    assert SyntheticISMParameters.__module__ == 'acoupipe.datasets.ism.parameters'


def test_non_conflicting_dataset_packages_own_config_classes():
    """Test that dataset Config classes live in dataset packages."""
    assert MIRACLESubmoduleConfig is MIRACLEConfig
    assert SRIRACHASubmoduleConfig is SRIRACHAConfig
    assert ISMSubmoduleConfig is SyntheticISMConfig
    assert MIRACLEConfig.__module__ == 'acoupipe.datasets.miracle.config'
    assert SRIRACHAConfig.__module__ == 'acoupipe.datasets.sriracha.config'
    assert SyntheticISMConfig.__module__ == 'acoupipe.datasets.ism.config'


def test_non_conflicting_dataset_packages_own_dataset_classes():
    """Test that dataset classes live in package-local dataset modules."""
    assert MIRACLESubmoduleDataset is DatasetMIRACLE
    assert SRIRACHASubmoduleDataset is DatasetSRIRACHA
    assert ISMSubmoduleDataset is DatasetSyntheticISM
    assert legacy_synthetic.DatasetSyntheticISM is DatasetSyntheticISM
    assert DatasetMIRACLE.__module__ == 'acoupipe.datasets.miracle.dataset'
    assert DatasetSRIRACHA.__module__ == 'acoupipe.datasets.sriracha.dataset'
    assert DatasetSyntheticISM.__module__ == 'acoupipe.datasets.ism.dataset'


def test_dataset_configs_create_dataset_specific_parameters():
    """Test that package configs default to their dataset-specific parameters."""
    assert isinstance(MIRACLEConfig().parameters, MIRACLEParameters)
    assert isinstance(SRIRACHAConfig().parameters, SRIRACHAParameters)
    assert isinstance(SyntheticISMConfig().parameters, SyntheticISMParameters)


def test_synthetic_package_owns_legacy_dataset_and_config_classes():
    """Test that the synthetic package replaces the old synthetic.py module."""
    assert hasattr(legacy_synthetic, '__path__')
    assert SyntheticSubmoduleDataset is DatasetSynthetic
    assert SyntheticLegacyConfig is DatasetSyntheticConfig
    assert ISMLegacyConfig is DatasetSyntheticISMConfig
    assert legacy_synthetic.DatasetSynthetic is DatasetSynthetic
    assert legacy_synthetic.DatasetSyntheticConfig is DatasetSyntheticConfig
    assert legacy_synthetic.DatasetSyntheticISMConfig is DatasetSyntheticISMConfig
    assert DatasetSynthetic.__module__ == 'acoupipe.datasets.synthetic.dataset'
    assert DatasetSyntheticConfig.__module__ == 'acoupipe.datasets.synthetic.legacy_config'
    assert DatasetSyntheticISMConfig.__module__ == 'acoupipe.datasets.ism.legacy_config'


def test_experimental_legacy_configs_live_in_dataset_packages():
    """Test that measured legacy configs live in their dataset packages."""
    assert MIRACLELegacyConfig.__module__ == 'acoupipe.datasets.miracle.legacy_config'
    assert SRIRACHALegacyConfig.__module__ == 'acoupipe.datasets.sriracha.legacy_config'


def test_synthetic_package_preserves_legacy_helper_imports():
    """Test that legacy helper functions remain importable from the package."""
    assert legacy_synthetic.sample_rms is synthetic_legacy_sample_rms
