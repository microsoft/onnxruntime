#!/bin/bash

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 

SERVER_URI=${1:-"localhost"}

echo "Waiting for TRITON Server to be ready at http://$SERVER_URI:8000..."

live_command="curl -i -m 1 -L -s -o /dev/null -w %{http_code} http://$SERVER_URI:8000/api/health/live"
ready_command="curl -i -m 1 -L -s -o /dev/null -w %{http_code} http://$SERVER_URI:8000/api/health/ready"

current_status=$($live_command)
echo $current_status

# First check the current status. If that passes, check the json. If either fail, loop
while [[ ${current_status} != "200" ]] || [[ $($ready_command) != "200" ]]; do

   printf "."
   sleep 1
   current_status=$($live_command)
done

echo "TRITON Server is ready!"
