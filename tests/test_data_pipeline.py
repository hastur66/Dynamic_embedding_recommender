from dynamic_embedding.data_pipeline import create_datasets

def test_create_datasets():    
    datasets = create_datasets()
    assert len(datasets.training_datasets.train_ds) == 220
    assert len(datasets.training_datasets.validation_ds) == 25