Due to an issue with how the Eigen release was downloaded the actual contents used in ORT 1.16 releases include some additional diffs beyond what is officially tagged as 3.4.0 in the Eigen repo.

See https://github.com/microsoft/onnxruntime/pull/18290 for the change to the Eigen package download.

In order to keep things consistent across all the ORT 1.16+ releases, the official Eigen 3.4.0 release is now downloaded, and 3.4.0_to_ORT_1.16_src.patch contains alls the diffs to make the source equivalent to what was previously used in the ORT builds.
 
This patch file includes the change that was originally applied by Fix_Eigen_Build_Break.patch. The patch file is kept to clarify changes that were not from official commits to the Eigen source.
