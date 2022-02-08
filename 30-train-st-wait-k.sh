#!/bin/bash

###
### ST training script for IWSLT 2022 speech-to-text baselines
###

basedir=`dirname $0`
scriptdir=${basedir}/scripts
FAIRSEQ_ROOT=${basedir}/fairseq
. ${scriptdir}/00-env.sh

parse_options "$@"
shift $?
if test ${verbose} -gt 0 ; then set -x ; fi

if test -z "${K}" ; then
    K=20
fi

if test -e ${WORKDIR}/ASR/${SRC}-${TRG}/.done ; then
    make_dir ${WORKDIR}/ST/${SRC}-${TRG}/wait-${K}
    env PYTHONPATH=${FAIRSEQ_ROOT} \
        ${python3} ${FAIRSEQ_ROOT}/train.py ${WORKDIR}/MuST-C/${SRC}-${TRG} \
            --load-pretrained-encoder-from ${WORKDIR}/ASR/${SRC}-${TRG}/checkpoint_best.pt \
            --config-yaml config_st.yaml \
            --train-subset train_st \
            --valid-subset dev_st \
            --save-dir ${WORKDIR}/ST/${SRC}-${TRG}/wait-${K} \
            --num-workers 4 \
            --max-tokens 40000 \
            --max-update 100000 \
            --task speech_to_text \
            --criterion label_smoothed_cross_entropy \
            --report-accuracy \
            --arch convtransformer_simul_trans_espnet \
            --simul-type waitk_fixed_pre_decision \
            --waitk-lagging ${K} \
            --fixed-pre-decision-ratio 7 \
            --optimizer adam \
            --lr 0.0001 \
            --lr-scheduler inverse_sqrt \
            --warmup-updates 4000 \
            --clip-norm 10.0 \
            --seed 1 \
            --update-freq 8
    if test $? -eq 0 ; then
        touch ${WORKDIR}/ST/${SRC}-${TRG}/wait-${K}/.done
    else
        error_and_die "Failed to train ST wait-${K}."
    fi
fi

exit 0
