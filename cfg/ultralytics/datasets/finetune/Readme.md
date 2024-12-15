all datasets: [train, val, test_synth, test_real_nominal, test_real_edge, test_real]

train on:
train

finetune on:
1. single: val
2. double: val + test_synth
3. triple: val + test_synth + test_real_nominal
4. triple_split: 2 + test_real_nominal

test on [using only 'last' weights]:
1. single: test_synth, test_real_nominal, test_real_edge, test_real, test
2. double: test_real_nominal, test_real_edge, test_real
3. triple: test_real_edge
4. triple_split: test_real_edge