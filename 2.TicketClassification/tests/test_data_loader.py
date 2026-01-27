def test_load_dataset(dataset):
    X_train, y_train, X_val, y_val, X_test, y_test, labels = dataset

    # Check basic lengths
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    # Check labels are consistent
    all_labels = set(y_train + y_val + y_test)
    for l in all_labels:
        assert l in labels

    # Check only top 5 labels are used
    assert len(labels) == 5


