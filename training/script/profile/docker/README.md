## Build ORT Docker Images for Philly Hypercluster ##

[Only the first time] Follow "Accessing Philly Images" in https://phillywiki.azurewebsites.net/articles/Custom_Job_V2.html to login Philly docker registry on your Linux machine.

Copy build.sh to your local folder. Modify the GIT_TOKEN, ONNXRUNTIME_SERVER_BRANCH, and Blob account key.

Run "docker run -it -v <Current Folder Path>:/scripts  phillyregistry.azurecr.io/philly/jobs/custom/onnxruntime:v1 bash /scripts/build.sh". It will pull specified branch, and detect latest commit, and build the binary with CUDA, then upload the images onto Azure Blob. (In Philly job, the ORT binary is downloaded before the benchmarking)