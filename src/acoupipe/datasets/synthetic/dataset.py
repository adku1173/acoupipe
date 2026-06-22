"""Legacy synthetic dataset class."""

from acoupipe.datasets.base import DatasetBase
from acoupipe.datasets.features import BaseFeatureCatalog, BaseFeatureCollectionBuilder
from acoupipe.datasets.synthetic.legacy_config import DatasetSyntheticConfig


class DatasetSynthetic(DatasetBase):
    r"""`DatasetSynthetic` is a purely synthetic microphone array source case generator.

    DatasetSynthetic relies on synthetic source signals from which the features are extracted and has been used in different publications,
    e.g. :cite:`Kujawski2019`, :cite:`Kujawski2022`, :cite:`Feng2022`. The default virtual simulation setup consideres a 64 channel microphone
    array and a planar observation area, as shown in the `default measurement setup`_ figure.

    **Default environmental properties**

    .. _Environmental Characteristics:

    .. table:: Default Environmental Characteristics

        ===================== ========================================
        Environment           Anechoic, Resting, Homogeneous Fluid
        Speed of sound        343 m/s
        Microphone Array      Vogel's spiral, :math:`M=64`, Aperture Size 1 m
        Observation Area      x,y in [-0.5,0.5], z=0.5
        Source Type           Monopole
        Source Signals        Uncorrelated White Noise (:math:`T=5\,s`)
        ===================== ========================================

    **Default FFT parameters**

    The underlying default FFT parameters are:

    .. table:: FFT Parameters

        ===================== ========================================
        Sampling Rate         He = 40, fs=13720 Hz
        Block size            128 Samples
        Block overlap         50 %
        Windowing             von Hann / Hanning
        ===================== ========================================


    **Default randomized properties**

    Several properties of the dataset are randomized for each source case when generating the data. Their respective distributions,
    are closely related to :cite:`Herold2017`. As such, the the microphone positions are spatially disturbed
    to account for uncertainties in the microphone placement. The number of sources, their positions, and strength is randomly chosen.
    Uncorrelated white noise is added to the microphone channels by default.

    .. table:: Randomized properties

        ==================================================================   ===================================================
        Sensor Position Deviation [m]                                        Bivariate normal distributed (:math:`\sigma = 0.001)`
        No. of Sources                                                       Poisson distributed (:math:`\lambda=3`)
        Source Positions [m]                                                 Bivariate normal distributed (:math:`\sigma = 0.1688`)
        Source Strength (:math:`[{Pa}^2]` at reference position)               Rayleigh distributed (:math:`\sigma_{R}=5`)
        Relative Noise Variance                                              Uniform distributed (:math:`10^{-6}`, :math:`0.1`)
        ==================================================================   ===================================================

    Example
    -------

    .. code-block:: python

        from acoupipe.datasets.synthetic import DatasetSynthetic

        dataset = DatasetSynthetic()
        dataset_generator = dataset.generate_dataset(
            features=['sourcemap', 'loc', 'f', 'num'],  # choose the features to extract
            f=[1000, 2000, 3000],  # choose the frequencies to extract
            split='training',  # choose the split of the dataset
            size=10,  # choose the size of the dataset
        )

        # get the first data sample
        data = next(dataset_generator)

        # print the keys of the dataset
        print(data.keys())


    **Initialization Parameters**
    """

    def __init__(
        self,
        mode='welch',
        mic_pos_noise=True,
        mic_sig_noise=True,
        snap_to_grid=False,
        random_signal_length=False,
        signal_length=5,
        fs=13720.0,
        min_nsources=1,
        max_nsources=10,
        tasks=1,
        remote_args=None,
        logger=None,
        config=None,
    ):
        """Initialize the DatasetSynthetic object.

        The input parameters are passed to the DatasetSyntheticConfig object, which creates
        all necessary objects for the simulation of microphone array data.

        Parameters
        ----------
        mode : str
            Type of calculation method. Can be either :code:`welch`, :code:`analytic` or :code:`wishart`.
            Defaults to :code:`welch`.
        mic_pos_noise : bool
            Apply positional noise to microphone geometry. Defaults to True.
        mic_sig_noise : bool
            Apply additional uncorrelated white noise to microphone signals. Defaults to True.
        snap_to_grid : bool
            Snap source locations to grid. The grid is defined in the config object as
            config.grid. Defaults to False.
        random_signal_length : bool
            Randomize signal length. Defaults to False. If True, the signal length is
            uniformly sampled from the interval [1s,10s].
        signal_length : float
            Length of the signal in seconds. Defaults to 5 seconds.
        fs : float
            Sampling frequency in Hz. Defaults to 13720 Hz.
        min_nsources : int
            Minimum number of sources in the dataset. Defaults to 1.
        max_nsources : int
            Maximum number of sources in the dataset. Defaults to 10.
        tasks : int
            Number of parallel tasks. Defaults to 1.
        remote_args : dict
            Dictionary of keyword arguments passed to the remote actors when using Ray for parallelization. Defaults to None.
        logger : logging.Logger
            Logger object. Defaults to None.
        config : DatasetSyntheticConfig
            Configuration object. Defaults to None. If None, a default configuration
            object is created.
        """
        if config is None:
            config = DatasetSyntheticConfig(
                mode=mode,
                signal_length=signal_length,
                fs=fs,
                min_nsources=min_nsources,
                max_nsources=max_nsources,
                mic_pos_noise=mic_pos_noise,
                mic_sig_noise=mic_sig_noise,
                snap_to_grid=snap_to_grid,
                random_signal_length=random_signal_length,
            )
        super().__init__(config=config, tasks=tasks, logger=logger, remote_args=remote_args)

    def get_feature_collection(self, features, f, num):
        """
        Get the feature collection of the dataset.

        Returns
        -------
        BaseFeatureCollection
            BaseFeatureCollection object.
        """
        # handle all custom features (BaseFeatureCatalog instances)
        custom_features = [feat for feat in features if isinstance(feat, BaseFeatureCatalog)]
        # collect default features defined by name
        default_feature_names = [feat for feat in features if isinstance(feat, str)]
        default_features = self.config.get_default_features(default_feature_names, f, num)
        builder = BaseFeatureCollectionBuilder(features=default_features + custom_features)
        builder.add_custom(self.config.get_prepare_func())  # add prepare function
        feature_collection = builder.build()  # finally build the feature collection
        builder.add_custom(self.config.get_cleanup_func(features))  # add cleanup function
        return feature_collection


__all__ = ['DatasetSynthetic']
