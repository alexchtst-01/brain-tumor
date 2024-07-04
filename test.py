# using data processing
from dataPreprocessing import CreateAnotationLayers

testingMigration = CreateAnotationLayers(
    src_path="raw_data/test",
    destination_path="data/test",
    anot_key='image_id'
)
testingMigration.createAnotation()

trainingMigration = CreateAnotationLayers(
    src_path="raw_data/train",
    destination_path="data/train",
    anot_key='image_id'
)
trainingMigration.createAnotation()

