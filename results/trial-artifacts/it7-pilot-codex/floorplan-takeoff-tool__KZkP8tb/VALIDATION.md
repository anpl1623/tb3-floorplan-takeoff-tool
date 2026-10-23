Validation completed on 23 October 2026 using fresh CLI processes restricted to two CPUs and running as uid/gid 65534.

| Supplied scan | Largest area error | Largest length error | Counts | Runtime |
| --- | ---: | ---: | --- | ---: |
| store-0486 | 0.295% | 0.015% | All exact | 16.20 s |
| store-0735 | 0.175% | 0.022% | All exact | 18.12 s |
| store-0828 | 0.279% | 0.027% | All exact | 18.15 s |

All individual quantities passed the requested absolute-or-relative tolerances. Percentages above are the maximum relative errors among individual finish or partition codes, rather than errors in their totals.

The final regression also passed 22 derived cases, for 25 cases in total:

- Each original was resampled to 65% resolution, faded with blur and noise, rotated by 1.25 degrees, given increased photocopy contrast and noise, turned upside down, and turned to portrait orientation.
- Three graphic-scale variants tested reading different numeric divisions and converting the resulting physical measurements.
- A modified plan with a four-foot by eight-foot exterior corner recess tested tracing a stepped perimeter and changing the floor area accordingly.

Across these 25 cases, the largest area error was 0.934%, the largest length error was 0.124%, every count was exact, and the slowest run took 19.40 seconds. Derived scans test specific robustness properties; they are not independent unseen stores.

Additional checks verified floor classification against the current sheet's legend with finish-code annotation recognition disabled; circulation boundaries opening toward all four sides; and swing recognition for 2.5, 2.667, 3.5, and 4-foot leaves.

An isolated deployment check copied only this package to a read-only directory, renamed the input, changed to a separate working directory, dropped privileges to uid/gid 65534, restricted CPU affinity to two CPUs, and blocked socket creation and connection with seccomp. The CLI completed in 16.59 seconds with all quantities passing and all counts exact. No examples directory, writable installation, network access, or runtime font files were needed.

The three supplied pairs can be checked again with:

```sh
python3 /app/takeoff/validate.py /app/examples --unprivileged
```
