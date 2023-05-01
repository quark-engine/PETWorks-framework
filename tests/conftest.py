import pytest

DATASET_PATHS = [
    {
        "originalData": "data/adult.csv",
        "anonymizedData": "data/adult_anonymized.csv",
        "dataHierarchy": "data/adult_hierarchy",
    },
    {
        "originalData": "data/delta.csv",
        "anonymizedData": "data/delta_anonymized.csv",
        "dataHierarchy": "data/delta_hierarchy",
    },
]


@pytest.fixture(scope="session")
def DATASET_PATH_ADULT():
    return DATASET_PATHS[0]


@pytest.fixture(scope="session")
def DATASET_PATH_DELTA():
    return DATASET_PATHS[1]
