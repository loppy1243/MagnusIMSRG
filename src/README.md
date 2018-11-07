Implementation of IMRSG with Magnus expansion in Julia, with the 4-particle 2-level pairing
model as an example.

Runs, but does not currently work.

# How to Run
In the top-level directory, open a Julia prompt and type
```Julia
julia> using Pkg
julia> Pkg.activate(".") # Switch the current project to the current directory
# Outputs "(path to current dir)/Project.toml"
julia> Pkg.instantiate() # Download dependencies
# Output varies
julia> include("run/main.jl") # Run
```
