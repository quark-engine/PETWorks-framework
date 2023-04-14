from PETWorks.arx import Data, loadDataFromCsv, JavaApi, UtilityMetrics


def _setDataHierarchies(data: Data, javaApi: JavaApi) -> None:
    for column in range(data.getHandle().getNumColumns()):
        data.getDefinition().setAttributeType(
            data.getHandle().getAttributeName(column),
            javaApi.Hierarchy.create(),
        )


def _measureAECS(original: Data, anonymized: Data) -> float:
    utility = (
        original.getHandle()
        .getStatistics()
        .getQualityStatistics(anonymized.getHandle())
    )

    aecs = utility.getAverageClassSize().getValue()
    return aecs


def PETValidation(original, anonymized, _):
    javaApi = JavaApi()
    original = loadDataFromCsv(
        original, javaApi.StandardCharsets.UTF_8, ";", javaApi
    )
    anonymized = loadDataFromCsv(
        anonymized, javaApi.StandardCharsets.UTF_8, ";", javaApi
    )

    _setDataHierarchies(original, javaApi)
    _setDataHierarchies(anonymized, javaApi)

    aecs = _measureAECS(original, anonymized, javaApi)
    return {"AECS": aecs}
