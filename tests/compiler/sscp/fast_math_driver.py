"""Driver-only checks; also runnable without a compiler build using Python 3."""
import pathlib
import runpy
from types import SimpleNamespace

source = pathlib.Path(__file__).resolve().parents[3] / "bin" / "acpp"
driver = runpy.run_path(str(source))
option = "-acpp-sscp-kernel-opts=fast-math"


def check(arguments, expected):
    config = SimpleNamespace(
        forwarded_compiler_arguments=arguments,
        has_plugin_sscp_compiler=True,
        clang_path="clang++",
        is_export_all=False,
        is_plugin_linked_into_llvm=True,
    )
    original = arguments[:]
    flags = driver["llvm_sscp_invocation"](config, []).get_cxx_flags()
    assert (option in flags) == expected, (arguments, flags, expected)
    assert arguments == original


for args, expected in [
    ([], False),
    (["-O3"], False),
    (["-fno-fast-math"], False),
    (["-ffinite-math-only"], False),
    (["-ffast-math"], True),
    (["-Ofast"], True),
    (["-Ofast", "-O3"], False),
    (["-O3", "-Ofast"], True),
    (["-ffast-math", "-Ofast", "-O3"], True),
    (["-Ofast", "-fno-fast-math"], False),
    (["-fno-fast-math", "-Ofast"], True),
    (["-O3", "-ffast-math", "-fno-fast-math"], False),
    (["-ffast-math", "-fno-fast-math", "-ffast-math"], True),
    (["-Ofast", "-fno-fast-math", "-Ofast"], True),
    (["-ffast-math", "-fhonor-nans", "-fhonor-infinities",
      "-fno-honor-nans"], False),
]:
    check(args, expected)

for cancel, restore in [
    ("-fhonor-nans", "-fno-honor-nans"),
    ("-fhonor-infinities", "-fno-honor-infinities"),
    ("-fno-finite-math-only", "-ffinite-math-only"),
    ("-fsigned-zeros", "-fno-signed-zeros"),
    ("-ftrapping-math", "-fno-trapping-math"),
    ("-frounding-math", "-fno-rounding-math"),
    ("-fmath-errno", "-fno-math-errno"),
    ("-fno-approx-func", "-fapprox-func"),
    ("-fno-associative-math", "-fassociative-math"),
    ("-fno-reciprocal-math", "-freciprocal-math"),
    ("-fno-unsafe-math-optimizations", "-funsafe-math-optimizations"),
    ("-ffp-contract=off", "-ffp-contract=fast"),
    ("-ffp-contract=on", "-ffp-contract=fast"),
    ("-ffp-exception-behavior=strict", "-ffp-exception-behavior=ignore"),
    ("-ffp-exception-behavior=maytrap", "-ffp-exception-behavior=ignore"),
]:
    for enable in ["-ffast-math", "-Ofast"]:
        check([enable, cancel], False)
        check([cancel, enable], True)
        check([enable, cancel, enable], True)
        check([enable, cancel, restore], True)

for model in ["precise", "strict", "fast", "aggressive"]:
    check(["-ffast-math", "-ffp-model=" + model], False)
    check(["-ffp-model=" + model, "-ffast-math"], True)

for name in ["-fdenormal-fp-math", "-fdenormal-fp-math-f32"]:
    check(["-ffast-math", name + "=ieee"], False)
    check([name + "=ieee", "-ffast-math"], False)
    check(["-ffast-math", name + "=ieee", name + "=preserve-sign"], True)

print("SSCP fast-math driver checks passed")
