# Running the Tests

To run all tests:
```shell
pytest tests
```

We use pytest and some of the tests are automatically generated.
You can run a subset using pytests built in features to filter by the matches on the name of the testcase using the -k argument.

For example to find tests relating to the trace operation, you can run
```shell
pytest tests -k trace
```

The usual pytest command line arguments apply (like -v for verbose).