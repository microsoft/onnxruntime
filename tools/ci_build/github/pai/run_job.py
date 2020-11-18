#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
 
import socket

def main():
    print('Hello from {}!'.format(socket.gethostname()))

if __name__ == "__main__":
    main()
