ncnn.patch is required to build.

record_download.patch was used during experimentation where we used DataTransferManager to copy to/from NCNN.
As NCNN has internal packing logic it was somewhat of a hack to replicate that.
Would most likely be required if we need to use DataTransferManager again.
