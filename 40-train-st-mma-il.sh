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

if test -e ${WORKDIR}/ASR/.done ; then
    make_dir ${WORKDIR}/ST/${SRC}-${TRG}/mma-il

    env PYTHONPATH=${FAIRSEQ_ROOT} \
        ${python3} ${FAIRSEQ_ROOT}/train.py ${WORKDIR}/MuST-C/${SRC}-${TRG} \
            --load-pretrained-encoder-from ${WORKDIR}/ASR/${SRC}-${TRG}/checkpoint_best.pt \
            --config-yaml config_st.yaml \
            --train-subset train_st \
            --valid-subset dev_st \
            --save-dir ${WORKDIR}/ST/${SRC}-${TRG}/mma-il \
            --num-workers 4 \
            --max-tokens 40000 \
            --max-update 100000 \
            --task speech_to_text \
            --criterion latency_augmented_label_smoothed_cross_entropy \
            --latency-weight-avg 0.1 \
            --arch convtransformer_simul_trans_espnet \
            --simul-type infinite_lookback_fixed_pre_decision \
            --fixed-pre-decision-ratio 7 \
            --optimizer adam \
            --lr 0.0001 \
            --lr-scheduler inverse_sqrt \
            --warmup-updates 4000 \
            --clip-norm 10.0 \
            --seed 1 \
            --update-freq 8
    if test $? -eq 0 ; then
        touch ${WORKDIR}/ST/${SRC}-${TRG}/mma-il/.done
    else
        error_and_die "Failed to train ST mma-il."
    fi
fi

exit 0
