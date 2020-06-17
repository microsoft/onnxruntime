add_library(mlperf_loadgen loadgen.h query_sample.h query_sample_library.h system_under_test.h test_settings.h
        loadgen.cc logging.cc logging.h test_settings_internal.cc test_settings_internal.h utils.cc utils.h version.h
        version.cc)