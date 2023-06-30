Initial timing data

```
$ for n in 16 32 64; do echo $n; julia --project=. checkerboard_example.jl $n 100000; done
16
args = ["16", "100000"]
Time per step (us) 0.5779123291232913
32
args = ["32", "100000"]
Time per step (us) 2.278357423574236
64
args = ["64", "100000"]
Time per step (us) 9.432052560525605
```
