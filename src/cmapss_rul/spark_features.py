"""PySpark implementation of the feature pipeline, for large fleets and Microsoft Fabric.

It produces the same columns as :func:`cmapss_rul.features.temporal_features`, which
a test asserts on real data. In Fabric the input is typically a Lakehouse Delta table
of raw telemetry and the output is written back as a feature table.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .data import SENSORS, SETTINGS


def regime_column(df: DataFrame) -> DataFrame:
    """Same discrete regime key as the pandas version (rounded settings)."""
    return df.withColumn(
        "regime",
        F.concat_ws(
            "|",
            F.format_number(F.round(F.col(SETTINGS[0]), 0), 1),
            F.format_number(F.round(F.col(SETTINGS[1]), 1), 1),
            F.format_number(F.round(F.col(SETTINGS[2]), 0), 1),
        ),
    )


def fit_regime_stats(train: DataFrame) -> DataFrame:
    """Per-regime mean and population std of every sensor (training data only)."""
    aggs = []
    for s in SENSORS:
        aggs += [F.avg(s).alias(f"{s}__mu"), F.stddev_pop(s).alias(f"{s}__sd")]
    return regime_column(train).groupBy("regime").agg(*aggs)


def normalize(df: DataFrame, stats: DataFrame, min_std: float = 1e-4) -> DataFrame:
    joined = regime_column(df).join(F.broadcast(stats), on="regime", how="inner")
    for s in SENSORS:
        sd = F.col(f"{s}__sd")
        safe_sd = F.when((sd < min_std) | sd.isNull(), F.lit(1.0)).otherwise(sd)
        joined = joined.withColumn(s, (F.col(s) - F.col(f"{s}__mu")) / safe_sd)
    return joined.drop("regime", *[c for c in joined.columns if "__" in c])


def temporal_features(df: DataFrame, sensors: list[str], window: int = 30) -> DataFrame:
    """Rolling mean / std / slope over the trailing ``window`` cycles of each unit."""
    w = Window.partitionBy("unit").orderBy("cycle").rowsBetween(-(window - 1), 0)
    t = F.col("cycle").cast("double")
    n = F.count(F.lit(1)).over(w)
    st = F.sum(t).over(w)
    st2 = F.sum(t * t).over(w)
    denom = n * st2 - st * st
    cols = [F.col("unit"), F.col("cycle")]
    for s in sensors:
        y = F.col(s)
        sy = F.sum(y).over(w)
        sty = F.sum(t * y).over(w)
        sy2 = F.sum(y * y).over(w)
        mean = sy / n
        var = F.greatest(sy2 / n - mean * mean, F.lit(0.0))
        slope = F.when(denom > 0, (n * sty - st * sy) / denom).otherwise(F.lit(0.0))
        cols += [mean.alias(f"{s}_mean"), F.sqrt(var).alias(f"{s}_std"), slope.alias(f"{s}_slope")]
    if "rul" in df.columns:
        cols.append(F.col("rul"))
    return df.select(*cols)
