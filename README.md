Example showing how to use NIXL with ray

## Run python example

```shell
python nixl_example_ray.py
```

Above script will launch 2 ray actors that exchange data using NIXL.

## Build docker image

```shell
docker build .
```

## Run benchmark

```shell
python benchmark.py
```

Example results:
```
+-----------+------------+-------------+---------------+--------------+
| use_ray   |   duration |   bandwidth |   concurrency |   num_trials |
+===========+============+=============+===============+==============+
| N         |    3.90303 |   0.120599  |             1 |            1 |
+-----------+------------+-------------+---------------+--------------+
| Y         |    6.33579 |   0.0742928 |             1 |            1 |
+-----------+------------+-------------+---------------+--------------+
```
