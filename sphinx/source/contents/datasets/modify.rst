.. _quickstart:

Modifying a dataset
===================

In some cases, you may want to modify a dataset. For example, you may want to use a different microphone geometry or want a different environment. 
Let's say you want to use the UMA 16 channel microphone array, a popular MEMS microphone array. The geometric layout is available in the Acoular_ package.

.. code-block:: python

    import acoular
    from pathlib import Path
    import matplotlib.pyplot as plt

    uma_file = Path(acoular.__file__).parent / 'xml' / 'minidsp_uma16.xml'
    mg = acoular.MicGeom(from_file=uma_file)

    plt.figure()
    plt.scatter(mg.mpos[0], mg.mpos[1])


Changing the underlying dataset configuration
---------------------------------------------

Each dataset in the :module:`acoupipe.datasets` module has its own default configuration object, which after instantiation is available as the `config` attribute.
The configuration object holds all necessary objects needed to generate the dataset. For example, the microphone geometry is stored in the `mics` attribute.

.. code-block:: python

    from acoupipe.datasets.synthetic import DatasetSynthetic1
    
    dataset = DatasetSynthetic1()
    print(dataset.config)

    plt.figure()
    plt.scatter(dataset.config.mics.mpos[0], dataset.config.mics.mpos[1]) 


Theoretically, one could change the microphone geometry by directly modifying the `mics` attribute. However, this is not recommended, as the configuration object is not aware of the change. 
A better way is to subclass a new configuration object and overwrite the method responsible for creating the microphone geometry. In this case, the method is called `create_mics`. The method is called during the instantiation of the dataset, so we need to overwrite it before we create the dataset. 


.. code-block:: python

    from acoupipe.datasets.synthetic import Dataset1Config

    class ConfigUMA(Dataset1Config):
        def _create_mics(self):
            uma_file = Path(acoular.__file__).parent / 'xml' / 'minidsp_uma16.xml'
            return acoular.MicGeom(from_file=uma_file)
    

With the new configuration object, we can create a new dataset that uses the UMA 16 microphone array.

.. code-block:: python

    config = ConfigUMA()
    dataset_uma = DatasetSynthetic1(config=config)

    plt.figure()
    plt.scatter(dataset.config.mics.mpos[0], dataset.config.mics.mpos[1])
