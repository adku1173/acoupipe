from traits.api import Either, Enum, HasPrivateTraits

NUMPY_FLOAT_DTYPES = Enum(("float64", "float32"))
NUMPY_COMPLEX_DTYPES = Enum(("complex128", "complex64"))
NUMPY_INT_DTYPES = Enum(("int64", "int8", "int16", "int32"))
NUMPY_UINT_DTYPES = Enum(("uint64", "uint8", "uint16", "uint32"))

TF_FLOAT_DTYPES = Enum(("float32", "float16", "float64"))
TF_COMPLEX_DTYPES = Enum(("complex64", "complex128"))
TF_INT_DTYPES = Enum(("int32", "int8", "int16", "int64"))
TF_UINT_DTYPES = Enum(("uint32", "uint8", "uint16", "uint64"))

class PrecisionConfig(HasPrivateTraits):

    float = Either((None, NUMPY_FLOAT_DTYPES))
    complex = Either(None, NUMPY_COMPLEX_DTYPES)
    int = Either(None, NUMPY_INT_DTYPES)
    uint = Either(None, NUMPY_UINT_DTYPES)
    tf_float = Either(None, TF_FLOAT_DTYPES)
    tf_complex = Either(None, TF_COMPLEX_DTYPES)
    tf_int = Either(None, TF_INT_DTYPES)
    tf_uint = Either(None, TF_UINT_DTYPES)

precision = PrecisionConfig()
