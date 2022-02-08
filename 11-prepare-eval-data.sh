#!/bin/bash

###
### Data preprocessing for IWSLT 2022 speech-to-text baselines
###

basedir=`dirname $0`
scriptdir=${basedir}/scripts
FAIRSEQ_ROOT=${basedir}/fairseq
. ${scriptdir}/00-env.sh

parse_options "$@"
shift $?
if test ${verbose} -gt 0 ; then set -x ; fi

make_dir ${WORKDIR}/MuST-C/${SRC}-${TRG}

for task in tst-COMMON tst-HE ; do
    if test ! -e ${WORKDIR}/MuST-C/${SRC}-${TRG}/eval-${task}/.done ; then
        make_dir ${WORKDIR}/MuST-C/${SRC}-${TRG}/eval-${task}
        env PYTHONPATH=${FAIRSEQ_ROOT} \
            ${python3} ${FAIRSEQ_ROOT}/examples/speech_to_text/seg_mustc_data_v2.py --data-root ${MUSTC_ROOT} --lang ${TRG} --output ${WORKDIR}/MuST-C/${SRC}-${TRG}/eval-${task} --task st --split ${task}
        if test $? -eq 0 ; then
            touch ${WORKDIR}/MuST-C/${SRC}-${TRG}/eval-${task}/.done
        else
            error_and_die "Failed to preprocess MuST-C dataset."
        fi
    fi
done

exit 0
