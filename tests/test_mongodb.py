import pytest
from src.database.mongodb import build_run_document, validate_run_document, MongoRunStore

def test_build_run_document_structure():
    doc = build_run_document(
        algorithm='test_algo',
        problem='test_problem',
        dataset='test_dataset',
        seed=42,
        params={'pop_size': 10},
        metrics={'accuracy': 0.95},
        evaluations=100,
        convergence=[0.1, 0.5, 0.95],
        time_s=1.5,
        experiment_id='unit_test_exp',
    )
    assert validate_run_document(doc) is True
    assert doc['algorithm'] == 'test_algo'
    assert doc['metrics']['time_s'] == 1.5

def test_validate_run_document_missing_fields():
    with pytest.raises(ValueError):
        validate_run_document({'algorithm': 'only_one_field'})

def test_safe_uri_masks_password():
    store = MongoRunStore(uri='mongodb://myuser:secret123@cluster0.fhionxu.mongodb.net:27017/test')
    assert 'secret123' not in store.safe_uri()
    assert '***:***@' in store.safe_uri()
