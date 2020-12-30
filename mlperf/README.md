# MLPerf HugeCTR

## Test commands

To test our MLPerf version of HugeCTR on a bare metal DGX A100, run the following commands:
```bash
# build docker image:
docker build -f mlperf/Dockerfile -t hugectr_mlperf .
# run the docker image:
docker run -it --rm --runtime=nvidia --privileged -v /mnt/:/mnt -v /raid:/raid hugectr_mlperf bash
# run the training
huge_ctr --train mlperf/configs/55296_8gpus.json
```

Please make sure to use this exact dockerfile under: `mlperf/Dockerfile` to test your changes. We have experienced cases when using a different Dockerfile or different base image yielded vastly different accuracy results.

## Expected results
At the moment you don't need to test any other configs except for the ones for batch size = 55296. This analysis only applies to these configs. Currently, this batch size presents a good trade-off between steps-to-converge and batch_size. This might change in the future.

The target is to achieve AUC=0.8025. Currently, 99% experiments achieve this target. The other 1% fails with a NaN loss early in the training. This is acceptable by MLPerf rules.

Detailed results:
* 6% of experiments hit the target at epoch=0.95 (step=72010)
* 91% of experiments hit the target at epoch=0.90 (step=68220)
* 2% of experiments hit the target at epoch=0.85 (step=64430)
* 1% of experiments fails with a NaN loss

A subtle convergence bug could negatively influence this distribution. Because of this it's best to do at least 10 full training runs and see when they converge before merging in your changes.

Failing to reach the target, reaching the target later than expected or getting a NaN loss more often than expected indicate a convergence bug. Please note this applies only to the config files for batch size = 55296. Other configs will reach the target accuracy at different points.
