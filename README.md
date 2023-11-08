# What?

- Testing the OpenCL specification. Dunno may make a image encoder for hexer's [byte2img](https://github.com/Croxx/hexer)

# Navigating the Code

- Just run `make` or `make debug` for debug build.
- Headers are included, and used by default
- Requires a rutime and icl loader. Consult the Arch wiki. Basically install `ocl-icd`(`ocl-icd-opencl-dev` if on debian based distro), and
download a suitable runtime for your GPU (`opencl-rusticl-mesa` on `amd`, `opencl-nvidia`, on `nvidia`, and `intel-compute-runtim` on `intel`)
