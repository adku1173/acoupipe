"""Ray Data support: a pipeline that provides the samples as a :class:`ray.data.Dataset`.

The modules of this subpackage require Ray Data (``ray[data]``). They are only imported when a
Ray Data dataset is requested, so that other AcouPipe modules do not depend on Ray Data.
"""
