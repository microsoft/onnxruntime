import sys
import random
import unittest
import function_tests

if __name__ == '__main__':
    loader = unittest.TestLoader()

    test_classes = [function_tests.HttpJsonPayloadTests]

    test_suites = []
    for tests in test_classes:
        tests.hosting_app_path = sys.argv[1]
        tests.model_path = sys.argv[2]
        tests.test_data_path = sys.argv[3]
        tests.server_port = random.randint(30000, 50000)

        test_suites.append(loader.loadTestsFromTestCase(tests))

    suites = unittest.TestSuite(test_suites)
    runner = unittest.TextTestRunner(verbosity=2)

    results = runner.run(suites)
