#!/bin/bash

SEARCH_DIRS=$@

find $SEARCH_DIRS -name '*.cu'
find $SEARCH_DIRS -name '*.cpp' -o -name '*.cxx' -o -name '*.c' -o -name '*.cc'
find $SEARCH_DIRS -name '*.cuh'
find $SEARCH_DIRS -name '*.h' -o -name '*.hpp' -o -name '*.inc' -o -name '*.inl' -o -name '*.hxx' -o -name '*.hdl'
