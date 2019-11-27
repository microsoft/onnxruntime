## Build ORT Docker Images for Philly HyperCluster ##

Copy build.sh to your local folder. Modify the GIT_TOKEN, and Blob account key. If you have multiple commits to build, check batch_build.sh for examples.

Run "docker run -it -v <Current Folder Path>:/scripts onnxtraining.azurecr.io/philly/pt_ort_env:v1 bash /scripts/build.sh <branch name> [commit id]", commit id is optional. It will pull specified branch, and detect latest commit, and build the binary with CUDA, then upload the images onto Azure Blob. (In Philly job, the ORT binary is downloaded before the benchmarking)

Be noted: onnxtraining.azurecr.io/philly/pt_ort_env:v1 is a copy of phillyregistry.azurecr.io/philly/jobs/custom/onnxruntime:v1, in case you don't have philly access. The image did not contain any confidential information, so should be okay to put on Azure docker registry.

If you want to use the one in Philly docker registry, follow "Accessing Philly Images" in https://phillywiki.azurewebsites.net/articles/Custom_Job_V2.html to login Philly docker registry on your Linux machine.
