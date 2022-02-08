#!/bin/bash

###
### Train a wait-k model
###

basedir=`dirname $0`
scriptdir=${basedir}/scripts
FAIRSEQ_ROOT=${basedir}/fairseq
SIMULEVAL_ROOT=${basedir}/SimulEval
. ${scriptdir}/00-env.sh

parse_options "$@"
shift $?
if test ${verbose} -gt 0 ; then set -x ; fi

if test -z "${K}" ; then
    K=20
fi
model_dir=${WORKDIR}/ST/${SRC}-${TRG}/wait-${K}

decoder_options=""
if test ${TRG} = "ja" ; then
    decoder_options="${decoder_options} --sacrebleu-tokenizer ja-mecab"
    decoder_options="${decoder_options} --eval-latency-unit char --no-space"
fi

if test ! -s ${model_dir}/checkpoint_best.pt ; then
    error_and_die "No model checkpoints are found in ${model_dir}."
elif test ! -e ${model_dir}/.done ; then
    warn "The model training has not completed. The best intermediate checkpoint will be used."
fi

for dataset in tst-COMMON tst-HE ; do
    env PYTHONPATH="${FAIRSEQ_ROOT}:${SIMULEVAL_ROOT}" \
        ${python3} ${SIMULEVAL_ROOT}/bin/simuleval \
            --agent ${FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/agents/fairseq_simul_st.agent.py \
            --source ${WORKDIR}/MuST-C/${SRC}-${TRG}/eval-${task}/${task}.wav_list \
            --target ${WORKDIR}/MuST-C/${SRC}-${TRG}/eval-${task}/${task}.ja \
            --data-bin ${WORKDIR}/MuST-C/${SRC}-${TRG} \
            --model-path ${model_dir}/checkpoint_best.pt \
            --output ${model_dir}/${dataset} \
            --scores ${decoder_options} \
            --gpu
done
