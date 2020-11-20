import numpy as np
from pyspark.sql.types import IntegerType

from petastorm.codecs import ScalarCodec, NdarrayCodec
from petastorm.unischema import Unischema, UnischemaField

MnistSchema = Unischema('MnistSchema', [
    UnischemaField('idx', np.int_, (), ScalarCodec(IntegerType()), False),
    UnischemaField('digit', np.int_, (), ScalarCodec(IntegerType()), False),
    UnischemaField('image', np.uint8, (28, 28), NdarrayCodec(), False),
])