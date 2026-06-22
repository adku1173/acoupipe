"""MIRACLE dataset parameters."""

from acoupipe.datasets.base.parameters import ParameterSet

MIRACLE_SCENARIOS = ('A1', 'D1', 'A2', 'R2')


class MIRACLEParameters(ParameterSet):
    """
    Shallow sample parameters for MIRACLE measured-propagation datasets.

    The measured scenario is sample provenance and part of this ParameterSet.
    Dataset split and SRIR directory remain Config concerns. Scenario-derived
    runtime values such as speed of sound and microphone geometry are excluded
    from this slim ParameterSet.
    """

    available_scenarios = MIRACLE_SCENARIOS

    def __init__(
        self,
        scenario='A1',
        fs=32000,
        signal_length=5.0,
        source_locations=None,
        source_rms=None,
        snr_db=None,
        **parameters,
    ):
        super().__init__(
            scenario=scenario,
            fs=fs,
            signal_length=signal_length,
            source_locations=source_locations,
            source_rms=source_rms,
            snr_db=snr_db,
            **parameters,
        )


__all__ = ['MIRACLE_SCENARIOS', 'MIRACLEParameters']
