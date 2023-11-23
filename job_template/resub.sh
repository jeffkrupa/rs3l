#!/bin/bash


iname=$1
stripped_base=$(basename $iname)

WZ_ZZ=$(echo "$iname" | grep 'wz_zz' > /dev/null && echo "True" || echo "False")
echo WZ_ZZ $WZ_ZZ
if [[ "$WZ_ZZ" == "False" ]]; then
   inpath="/store/user/jkrupa/cl/infiles/mar20_finetuning_test"
else
   inpath="/store/user/jkrupa/cl/infiles/mar20/wz-vs-zz_test"
fi

sed "s|ZZZ|${iname}|g" environ_template.sh > ${iname}/environ.sh
sed -i "s|XXX|${stripped_base}|g" ${iname}/environ.sh
sed -i "s|AAA|${inpath}|g" ${iname}/environ.sh


source ${iname}/environ.sh

xrdfs root://xrootd.cmsaf.mit.edu:1094/ ls $inpath | awk -F/ '{print $NF}' > $iname/flat_infiles.txt
xrdfs root://eosuser.cern.ch/ mkdir /eos/project/c/contrast/public/cl/analysis/outfiles/${stripped_base}

jobdir="${stripped_base//,/__}"
rm -rf ../jobdir/${jobdir}
mkdir -p ../jobdir/${jobdir}

tar -cf ../jobdir/${jobdir}/myextrafiles.tar -C $iname FT_best-epoch.pt environ.sh
#tar -cf ../jobdir/${jobdir}/myextrafiles.tar $iname/FT_best-epoch.pt $iname/environ.sh

. get_missing.sh $iname/flat_infiles.txt > $iname/missing_files.txt 

tar czf deepjet-geometric.tgz ./deepjet-geometric
cp submit $iname/submit
sed -i "s|XXX|$(pwd)/../jobdir/${jobdir}|g" $iname/submit
sed -i "s|WWW|${iname}|g" $iname/submit

condor_submit $iname/submit
