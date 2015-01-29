# CUDA MD5 cracker
* Really simple
* Multi GPU support
* Any length (currently 1-8)
* Any charset (currently A-Za-z0-9)

# Benchmarks
* 1x GeForce GTX 660M  (w/  display attached) = **220 MHash/s**
* 2x GeForce GTX 460   (w/o display attached) = **1.65 GHash/s**
* 1x GeForce GTX Titan (w/  display attached) = **2.3 GHash/s**

# Compilation
* make

# Usage
```
$ ./md5_gpu 2a21e2561359ebf2fb2d634ee7837a8e
Notice: 2 device(s) found
Notice: currently at cKLtub (6)
Notice: cracked Nvidia
Notice: computation time 1327.7 ms
```

# Requirements
* make
* NVIDIA CUDA Compiler
* gcc or visual studio
* Most likely will work on any OS (Linux, Windows, and so on)

# Contribution
* Always welcome :)
