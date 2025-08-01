#!/bin/bash

URL=https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip
DIR=data/datasets/ecg/
FNAME=ecg.csv

if [ -f ${DIR}${FNAME} ]; then
    echo "ECG Dataset already exists"
    exit
fi

mkdir -p ${DIR}
cd ${DIR}
curl ${URL} --output ecg.zip
unzip ecg.zip

tail -n +10 ECG5000_TRAIN.ts > ${FNAME}
tail -n +10 ECG5000_TEST.ts >> ${FNAME}
sed -i -e 's/:/,/g; s/2$/1/; s/3$/1/; s/4$/1/; s/5$/1/' ${FNAME}
rm *.txt
rm *.ts
rm *.arff
rm ecg.zip
cd -
