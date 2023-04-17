# Tutorials

These tutorials are designed to do all the "important stuff",
    so that you can study the code and understand how to do it yourself.
They are also designed to be very modular,
    so that you can (hopefully) clone the module and just make a few tweaks
    to accomplish more system-specific tasks.

To study the code, start from the `run` function and go from there.



##  Optimization

This tutorial optimizes a system using a fixed number of evenly-spaced square windows.

To do the built-in run with H2 in the parity mapping:
```
]activate pkgs/Tutorials
import Tutorials: Optimization as OT
OT.run()
```

To do a run with a different matrix or parameters:
```
OT.modify_settings(
    matrixfile = "pathtomymatrixfile.npy",
    T = 120.0,
    W = 200,
)
OT.run(; subdir="whereIwanttosavedata")
```

To resume an optimization that was interrupted or had not yet converged:
```
OT.run(; subdir="pathtodata", loaddir=true, resume=true)
```

To take an optimization result from one directory and try optimizing with stricter constraints:
```
OT.load(; subdir="pathtodata", resume=false)
OT.modify_settings(
    g_tol = 1e-8,
)
OT.run_optimization("whereIwanttosavenewdata")
OT.update("whereIwanttosavenewdata")
```
