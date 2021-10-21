**Tests for global threadpools**

These tests here test the usage of the global threadpools using the C API data flow. The reason we need to create a 
separate exe here is because we need to create a separate environment that enables the creation of global threadpools.
The test environment used in the other exes create the env without global threadpools and since this env is process
wide (a singleton) we cannot use it for this kind of test.