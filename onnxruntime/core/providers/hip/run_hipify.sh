#! /bin/sh
for file in *
do
    if [ -f "$file"  ]; then
        newfile="$(echo ${file} |sed -e 's/hip/hip/')" ;
        /opt/hip/bin/hipify-perl $file > $newfile
   fi
done