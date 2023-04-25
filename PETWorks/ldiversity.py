from typing import Dict

import pandas as pd

from PETWorks.arx import (
    JavaApi,
    getDataFrame,
    loadDataFromCsv,
)
from PETWorks.attributetypes import QUASI_IDENTIFIER, SENSITIVE_ATTRIBUTE


def measureLDiversity(
        anonymizedData: pd.DataFrame,
        attributeTypes: Dict[str, str],
) -> list[int]:

    qis = []
    sensitiveAttributes = []
    lValues = []

    for attribute, value in attributeTypes.items():
        if value == QUASI_IDENTIFIER:
            qis.append(attribute)
        if value == SENSITIVE_ATTRIBUTE:
            sensitiveAttributes.append(attribute)

    for index in range(len(sensitiveAttributes)):
        columns = qis + sensitiveAttributes[: index] + sensitiveAttributes[index + 1:]
        groups = anonymizedData.groupby(columns)

        sensitiveAttribute = sensitiveAttributes[index]
        lValues += [
            group[sensitiveAttribute].nunique() for _, group in groups
        ]

    return lValues


def validateLDiversity(
        lValues: list[int], l: int
) -> bool:
    return all(value >= l for value in lValues)


def PETValidation(
        original, sample, _, attributeTypes, l
):
    javaApi = JavaApi()
    anonymizedData = loadDataFromCsv(
        sample, javaApi.StandardCharsets.UTF_8, ";", javaApi
    )

    anonymizedDataFrame = getDataFrame(anonymizedData)

    lValues = measureLDiversity(
        anonymizedDataFrame, attributeTypes
    )
    fulfillLDiversity = validateLDiversity(lValues, l)

    return {"l": l, "fulfill l-diversity": fulfillLDiversity}