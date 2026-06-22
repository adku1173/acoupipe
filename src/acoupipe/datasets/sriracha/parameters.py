"""SRIRACHA dataset parameters."""

from acoupipe.datasets.miracle.parameters import MIRACLEParameters

SRIRACHA_SCENARIOS = (
    'SR1',
    'SR1-C1',
    'SR1-C2',
    'SR1-C3',
    'SR1-C4',
    'SR1-D',
    'SR2',
    'SR2-C1',
    'SR2-C2',
    'SR2-C3',
    'SR2-C4',
    'SR2-D',
    'SRA1',
    'SRA1-C1',
    'SRA1-C2',
    'SRA1-C3',
    'SRA1-C4',
    'SRA1-D',
    'SRA2',
    'SRA2-C1',
    'SRA2-C2',
    'SRA2-C3',
    'SRA2-C4',
    'SRA2-D',
)


class SRIRACHAParameters(MIRACLEParameters):
    """
    Shallow sample parameters for SRIRACHA measured-propagation datasets.

    SRIRACHA uses the same first-pass ParameterSet schema as
    :class:`MIRACLEParameters`, with a SRIRACHA-specific scenario vocabulary and
    default scenario.
    """

    available_scenarios = SRIRACHA_SCENARIOS

    def __init__(self, scenario='SR1-D', **parameters):
        super().__init__(scenario=scenario, **parameters)


__all__ = ['SRIRACHA_SCENARIOS', 'SRIRACHAParameters']
