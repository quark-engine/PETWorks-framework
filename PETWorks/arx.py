import re
import pandas as pd
import numpy as np
from os import PathLike, listdir
from os.path import join
from typing import Dict, Iterator, List, Tuple
from PETWorks.attributetypes import IDENTIFIER, INSENSITIVE_ATTRIBUTE

from py4j.java_gateway import JavaGateway, JavaClass
from py4j.protocol import Py4JJavaError


PATH_TO_ARX_LIBRARY = "arx/lib/libarx-3.9.0.jar"

Data = JavaClass
Charset = JavaClass
StandardCharsets = JavaClass
CSVHierarchyInput = JavaClass
Hierarchy = JavaClass
ARXConfiguration = JavaClass
KAnonymity = JavaClass
ARXAnonymizer = JavaClass
ARXResult = JavaClass
AttributeType = JavaClass
Int = JavaClass

javaApiTable = {
    "Data": "jvm.org.deidentifier.arx.Data",
    "Charset": "jvm.java.nio.charset.Charset",
    "StandardCharsets": "jvm.java.nio.charset.StandardCharsets",
    "CSVHierarchyInput": "jvm.org.deidentifier.arx.io.CSVHierarchyInput",
    "Hierarchy": "jvm.org.deidentifier.arx.AttributeType.Hierarchy",
    "DefaultHierarchy": "jvm.org.deidentifier.arx.AttributeType.Hierarchy.DefaultHierarchy",
    "ARXConfiguration": "jvm.org.deidentifier.arx.ARXConfiguration",
    "KAnonymity": "jvm.org.deidentifier.arx.criteria.KAnonymity",
    "ARXAnonymizer": "jvm.org.deidentifier.arx.ARXAnonymizer",
    "AttributeType": "jvm.org.deidentifier.arx.AttributeType",
    "ARXPopulationModel": "jvm.org.deidentifier.arx.ARXPopulationModel",
    "DataHandle": "jvm.org.deidentifier.arx.DataHandle",
    "DataSubset": "jvm.org.deidentifier.arx.DataSubset",
    "HashGroupifyEntry": "jvm.org.deidentifier.arx.framework.check.groupify.HashGroupifyEntry",
    "HashSet": "jvm.java.util.HashSet",
    "DPresence": "jvm.org.deidentifier.arx.criteria.DPresence",
    "Int": "jvm.int",
    "String": "jvm.java.lang.String",
    "new_array": "new_array",
}


def createJavaGateway() -> JavaGateway:
    return JavaGateway.launch_gateway(classpath=PATH_TO_ARX_LIBRARY)


class JavaApi:
    def __init__(
        self,
        gatewayObject: JavaGateway = None,
        apiTable: Dict[str, str] = javaApiTable,
    ):
        self.gatewayObject = createJavaGateway()

        for name, javaApi in apiTable.items():
            api = eval("self.gatewayObject." + javaApi)
            setattr(self, name, api)



Data = gateway.jvm.org.deidentifier.arx.Data
Charset = gateway.jvm.java.nio.charset.Charset
CSVHierarchyInput = gateway.jvm.org.deidentifier.arx.io.CSVHierarchyInput
Hierarchy = gateway.jvm.org.deidentifier.arx.AttributeType.Hierarchy
ARXConfiguration = gateway.jvm.org.deidentifier.arx.ARXConfiguration
KAnonymity = gateway.jvm.org.deidentifier.arx.criteria.KAnonymity
ARXAnonymizer = gateway.jvm.org.deidentifier.arx.ARXAnonymizer
AttributeType = gateway.jvm.org.deidentifier.arx.AttributeType
Int = gateway.jvm.int



def loadDataFromCsv(
    path: PathLike, charset: Charset, delimiter: str, javaApi: JavaApi
) -> Data:
    return javaApi.Data.create(path, charset, delimiter)


def __findHierarchyFile(path: PathLike) -> Iterator[Tuple[str, PathLike]]:
    for filename in listdir(path):
        result = re.match(".*hierarchy_(.*?).csv", filename)
        if result is None:
            continue

        attributeName = result.group(1)

        dataHierarchyFile = join(path, filename)

        yield attributeName, dataHierarchyFile


def loadDataHierarchy(
    path: PathLike, charset: Charset, delimiter: str, javaApi: JavaApi
) -> Dict[str, Hierarchy]:
    return {
        attributeName: javaApi.Hierarchy.create(
            javaApi.CSVHierarchyInput(
                hierarchyFile, charset, delimiter
            ).getHierarchy()
        )
        for attributeName, hierarchyFile in __findHierarchyFile(path)
    }


def loadDataHierarchyNatively(
    path: PathLike, delimiter: str
) -> Dict[str, np.chararray]:
    return {
        attributeName: pd.read_csv(hierarchyFile, sep=delimiter).to_numpy(
            dtype=str
        )
        for attributeName, hierarchyFile in __findHierarchyFile(path)
    }


def setDataHierarchies(
    data: Data,
    hierarchies: Dict[str, Hierarchy],
    attributeTypes: Dict[str, str],
    javaApi: JavaApi,
) -> None:
    for attributeName, hierarchy in hierarchies.items():
        data.getDefinition().setAttributeType(attributeName, hierarchy)
        attributeType = attributeTypes.get(attributeName)

        if attributeType == IDENTIFIER:
            data.getDefinition().setAttributeType(
                attributeName, javaApi.AttributeType.IDENTIFYING_ATTRIBUTE
            )

        if attributeType == INSENSITIVE_ATTRIBUTE:
            data.getDefinition().setAttributeType(
                attributeName, javaApi.AttributeType.INSENSITIVE_ATTRIBUTE
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
    anonymizedSubset: Data, hierarchies: dict[str, list[list[str]]]
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



def convertJavaListToList(javaList) -> Tuple:
    length = len(javaList)
    return tuple(javaList[index] for index in range(length))


def anonymizeData(
    original: Data,
    privacyModels: List[str],
    javaApi: JavaApi,
    utilityModel: str = None,
    suppressionLimit: float = 0.0,
) -> ARXResult:
    arxConfig = javaApi.ARXConfiguration.create()

    arxConfig.setSuppressionLimit(suppressionLimit)

    for privacyModel in privacyModels:
        arxConfig.addPrivacyModel(privacyModel)

    if utilityModel:
        arxConfig.setQualityModel(utilityModel)

    try:
        anonymizer = javaApi.ARXAnonymizer()
        anonymizedData = anonymizer.anonymize(original, arxConfig)
    except Py4JJavaError as e:
        raise e

    return anonymizedData


def applyAnonymousLevels(
    original: Data,
    anonymousLevels: List[int],
    hierarchies: Dict[str, Hierarchy],
    attributeTypes: Dict[str, str],
    javaApi: JavaApi,
) -> Data:
    levels = javaApi.new_array(javaApi.Int, len(anonymousLevels))
    for i in range(len(anonymousLevels)):
        levels[i] = anonymousLevels[i]

    privacyModels = [javaApi.KAnonymity(1)]

    try:
        anonymizedData = anonymizeData(original, privacyModels, javaApi)
    except Py4JJavaError:
        return

    lattice = anonymizedData.getLattice()
    node = lattice.getNode(levels)

    result = javaApi.Data.create(
        anonymizedData.getOutput(node, True).iterator()
    )

    setDataHierarchies(result, hierarchies, attributeTypes, javaApi)

    return result
