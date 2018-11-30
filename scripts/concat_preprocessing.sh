# concatenation of .share files, generated
# for preprocessing
#
# this is useful paired with running generation in parallel
#     seq $TASKS | xargs -i -n 1 -P $CORES python scripts/generate_preprocessing {}
#
# The inputs are read from sharedata/
# The outputs are put in concatsharedata/

N=16
t=$(expr \( $N - 1 \) / 3)
k=4096
set -e
set -x
for kind in "inputs" "triples" "bits"
do
    BASE="sharedata/butterfly-N${N}-k${k}-${kind}"
    for party in $(seq 0 $(expr $N - 1))
    do
	DST="concat$BASE-${party}.share"
	echo DST: $DST
	echo "52435875175126190479447740508185965837690552500527637822603658699938581184513" > $DST
	echo $t >> $DST
	echo $party >> $DST
	for tag in $(seq 32)
	do
	    SRC="$BASE-${tag}-${party}.share"
	    cat $SRC | tail -n+4 >> $DST
	done
    done
done
