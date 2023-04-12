import re
import pandas as pd
import numpy as np
from os import PathLike, listdir
from os.path import join
from typing import List
from typing import Dict
from PETWorks.attributetypes import IDENTIFIER, INSENSITIVE_ATTRIBUTE, QUASI_IDENTIFIER, SENSITIVE_ATTRIBUTE

from py4j.java_gateway import JavaGateway, get_field

PATH_TO_ARX_LIBRARY = "arx/lib/libarx-3.9.0.jar"
gateway = JavaGateway.launch_gateway(
    classpath=PATH_TO_ARX_LIBRARY, die_on_exit=True
)

Data = gateway.jvm.org.deidentifier.arx.Data
Charset = gateway.jvm.java.nio.charset.Charset
CSVHierarchyInput = gateway.jvm.org.deidentifier.arx.io.CSVHierarchyInput
Hierarchy = gateway.jvm.org.deidentifier.arx.AttributeType.Hierarchy
ARXConfiguration = gateway.jvm.org.deidentifier.arx.ARXConfiguration
KAnonymity = gateway.jvm.org.deidentifier.arx.criteria.KAnonymity
ARXAnonymizer = gateway.jvm.org.deidentifier.arx.ARXAnonymizer
AttributeType = gateway.jvm.org.deidentifier.arx.AttributeType
Int = gateway.jvm.int

def loadDataFromCsv(path: PathLike, charset: Charset, delimiter: str) -> Data:
    return Data.create(path, charset, delimiter)


def loadDataHierarchy(
    path: PathLike, charset: Charset, delimiter: str
) -> dict[str, List[List[str]]]:
    hierarchies = {}
    for filename in listdir(path):
        result = re.match(".*hierarchy_(.*?).csv", filename)
        if result is None:
            continue

        attributeName = result.group(1)

        dataHierarchyFile = join(path, filename)
        hierarchy = CSVHierarchyInput(
            dataHierarchyFile, charset, delimiter
        ).getHierarchy()

        hierarchies[attributeName] = Hierarchy.create(hierarchy)

    return hierarchies


def setDataHierarchies(
    data: Data,
    hierarchies: Dict[str, Hierarchy],
    attributeTypes: Dict[str, str],
    ) -> None:
    for attributeName, attributeType in attributeTypes.items():
        if attributeName in hierarchies.keys():
            data.getDefinition().setAttributeType(
                attributeName, hierarchies[attributeName]
            )

        else:
            attributeType = attributeTypes.get(attributeName)

            if attributeType == IDENTIFIER:
                javaAttributeType = (
                    AttributeType.IDENTIFYING_ATTRIBUTE
                )
            elif attributeType == QUASI_IDENTIFIER:
                javaAttributeType = (
                    AttributeType.QUASI_IDENTIFYING_ATTRIBUTE
                )
            elif attributeType == SENSITIVE_ATTRIBUTE:
                javaAttributeType = (
                    AttributeType.INSENSITIVE_ATTRIBUTE
                )
            elif attributeType == INSENSITIVE_ATTRIBUTE:
                javaAttributeType = (
                    AttributeType.INSENSITIVE_ATTRIBUTE
                )
            else:
                raise ValueError(
                    f"Unexpected attribute type: {attributeType}"
                )

            data.getDefinition().setAttributeType(
                attributeName, javaAttributeType
            )



def getQiNames(dataHandle: str) -> list[str]:
    qiNameSet = dataHandle.getDefinition().getQuasiIdentifyingAttributes()
    qiNames = [qiName for qiName in qiNameSet]
    qiNames.sort(key=dataHandle.getColumnIndexOf)
    return qiNames


def getQiIndices(dataHandle: str) -> list[int]:
    qiNames = getQiNames(dataHandle)
    qiIndices = []
    for qiName in qiNames:
        qiIndices.append(dataHandle.getColumnIndexOf(qiName))

    return qiIndices


def findAnonymousLevel(hierarchy: list[list[str]], value: str) -> int:
    for hierarchyRow in hierarchy:
        for level in range(len(hierarchyRow)):
            if hierarchyRow[level] == value:
                return level
    return -1


def getAnonymousLevels(
    anonymizedSubset: Data, hierarchies: dict[str, Hierarchy]
) -> list[int]:
    subsetDataFrame = getDataFrame(anonymizedSubset.getHandle())
    subsetRowNum = len(subsetDataFrame)

    qiIndices = getQiIndices(anonymizedSubset.getHandle())

    sampleRowIndex = -1
    allSuppressed = False
    for subsetRowIndex in range(subsetRowNum):
        for qiIndex in qiIndices:
            if subsetDataFrame.iloc[subsetRowIndex][qiIndex] != "*":
                sampleRowIndex = subsetRowIndex
                break

        if sampleRowIndex != -1:
            break

        allSuppressed = (subsetRowIndex == subsetRowNum - 1)

    anonymousLevels = []
    for qiIndex in qiIndices:
        value = subsetDataFrame.iloc[sampleRowIndex][qiIndex]
        attributeName = subsetDataFrame.columns[qiIndex]
        hierarchy = hierarchies[attributeName].getHierarchy()

        if allSuppressed:
            anonymousLevels.append(len(hierarchy[0]) - 1)
            continue

        anonymousLevels.append(findAnonymousLevel(hierarchy, value))

    return anonymousLevels


def getDataFrame(dataHandle: str) -> pd.DataFrame:
    rowNum = dataHandle.getNumRows()
    colNum = dataHandle.getNumColumns()

    data = []
    for rowIndex in range(rowNum):
        row = []
        for colIndex in range(colNum):
            row.append(dataHandle.getValue(rowIndex, colIndex))
        data.append(row)

    colNames = [
        dataHandle.getAttributeName(colIndex) for colIndex in range(colNum)
    ]

    return pd.DataFrame(data, columns=colNames)


def getSubsetIndices(
    table: str,
    subset: str,
) -> list[int]:
    qiNames = getQiNames(table)
    qiIndices = getQiIndices(table)

    tableDataFrame = getDataFrame(table)
    groupedSubset = getDataFrame(subset).groupby(qiNames)

    tableRowNum = len(tableDataFrame)

    subsetIndices = []
    for _, subsetGroup in groupedSubset:
        subsetGroupList = subsetGroup.values.tolist()
        filter = pd.Series(True, index=range(tableRowNum))
        for qiName, qiIndex in zip(qiNames, qiIndices):
            filter &= (tableDataFrame[qiName] == subsetGroupList[0][qiIndex])

        subsetIndices += np.flatnonzero(filter).tolist()[:len(subsetGroupList)]

    return subsetIndices


def applyAnonymousLevels(original: Data, anonymousLevels: list[int]) -> str:
    levels = gateway.new_array(Int, len(anonymousLevels))
    for i in range(len(anonymousLevels)):
        levels[i] = anonymousLevels[i]

    arxConfig = ARXConfiguration.create()
    arxConfig.addPrivacyModel(KAnonymity(1))

    anonymizer = ARXAnonymizer()
    result = anonymizer.anonymize(original, arxConfig)

    lattice = result.getLattice()
    node = lattice.getNode(levels)

    return result.getOutput(node, True)
