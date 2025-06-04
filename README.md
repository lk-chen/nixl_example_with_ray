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
+------------+------------+--------------------+----------------------+--------------------+
| With Ray   |   Time (s) |   Bandwidth (GB/s) |   Concurrent Readers |   Number of Trials |
+============+============+====================+======================+====================+
| N          |    4.11485 |          0.114391  |                    1 |                  1 |
+------------+------------+--------------------+----------------------+--------------------+
| Y          |    5.95003 |          0.0791094 |                    1 |                  1 |
+------------+------------+--------------------+----------------------+--------------------+
```
