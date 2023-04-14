
from PETWorks.arx import Data, loadDataFromCsv, loadDataHierarchy
from PETWorks.arx import JavaApi, UtilityMetrics, setDataHierarchies


def _measureNonUniformEntropy(original: Data, anonymized: Data) -> float:
    utility = (
        original.getHandle()
        .getStatistics()
        .getQualityStatistics(anonymized.getHandle())
    )
    nonUniformEntropy = utility.getNonUniformEntropy().getArithmeticMean(False)
    return nonUniformEntropy


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

    nonUniformEntropy = _measureNonUniformEntropy(original, anonymized)
    return {"Non-Uniform Entropy": nonUniformEntropy}
