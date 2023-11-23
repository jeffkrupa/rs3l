which=$1

rm -f running
rm -f tmp
#echo source ${which}/environ.sh

echo " " > running

xrdfs root://eosuser.cern.ch/ ls ${ODIR} | awk -F/ '{print $NF}' > tmp 
#sed -i 's/\///g' tmp
sed -i '/\.sys\.v#\./d' tmp

while read p;
do
    h5name="${p/root/h5}"
    if grep -Fxq "$h5name" tmp    
    then
        continue
    else
	if grep -Fxq "$p" running
	then 
	    continue
	else
            echo $p
	fi
    fi
done < $1

rm -f tmp
rm -f running
