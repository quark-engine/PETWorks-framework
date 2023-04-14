
from PETWorks.arx import Data, loadDataFromCsv, loadDataHierarchy
from PETWorks.arx import (
    JavaApi,
    UtilityMetrics,
    setDataHierarchies,
)


def _measurePrecision(original: Data, anonymized: Data) -> float:
    utility = (
        original.getHandle()
        .getStatistics()
        .getQualityStatistics(anonymized.getHandle())
    )

    precision = utility.getGeneralizationIntensity().getArithmeticMean(False)
    return precision


def PETValidation(original, anonymized, _, dataHierarchy, **other):
    javaApi = JavaApi()

    dataHierarchy = loadDataHierarchy(
        dataHierarchy, javaApi.StandardCharsets.UTF_8, ";", javaApi
    )

    original = loadDataFromCsv(
        original, javaApi.StandardCharsets.UTF_8, ";", javaApi
    )
    anonymized = loadDataFromCsv(
        anonymized, javaApi.StandardCharsets.UTF_8, ";", javaApi
    )

    setDataHierarchies(original, dataHierarchy, javaApi)
    setDataHierarchies(anonymized, dataHierarchy, javaApi)

    precision = _measurePrecision(original, anonymized)
    return {"precision": precision}
