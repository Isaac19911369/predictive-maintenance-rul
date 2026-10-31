"""The PySpark pipeline must produce the same features as the pandas pipeline."""

import shutil

import numpy as np
import pytest

pyspark = pytest.importorskip("pyspark")
if shutil.which("java") is None:
    pytest.skip("Java runtime not available", allow_module_level=True)

from pyspark.sql import SparkSession  # noqa: E402

from cmapss_rul import spark_features as sf  # noqa: E402
from cmapss_rul.features import RegimeNormalizer, temporal_features  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    s = (
        SparkSession.builder.master("local[1]")
        .appName("parity")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    yield s
    s.stop()


def test_spark_matches_pandas(spark, fleet):
    train, _, _ = fleet
    norm = RegimeNormalizer().fit(train)
    expected = temporal_features(norm.transform(train), norm.sensors, window=15)

    sdf = spark.createDataFrame(train)
    stats = sf.fit_regime_stats(sdf)
    got = (
        sf.temporal_features(sf.normalize(sdf, stats), norm.sensors, window=15)
        .toPandas()
        .sort_values(["unit", "cycle"])
        .reset_index(drop=True)
    )

    assert list(got.columns) == list(expected.columns)
    np.testing.assert_allclose(got.to_numpy(float), expected.to_numpy(float), atol=1e-6)
