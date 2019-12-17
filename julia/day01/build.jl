using PackageCompiler

# This command will use the `runtest.jl` of `ColorTypes` + `FixedPointNumbers` to find out what functions to precompile!
# `force = false` to not force overwriting Julia's current system image
# compile_package("ColorTypes", "FixedPointNumbers", force = false) 

# force = false is the default and recommended, since overwriting your standard system image can make Julia unusable.

# If you used force and want your old system image back (force will overwrite the default system image Julia uses) you can run:
#revert()

# Or if you simply want to get a native system image e.g. when you have downloaded the generic Julia install:
#force_native_image!()

# Build an executable
build_executable(
    "forloop_cpu.jl", # Julia script containing a `julia_main` function, e.g. like `examples/hello.jl`
    # snoopfile = "call_functions.jl", # Julia script which calls functions that you want to make sure to have precompiled [optional]
    builddir = "/work/builddir" # that's where the compiled artifacts will end up [optional]
)

# Build a shared library
build_shared_lib("forloop_cpu.jl")
