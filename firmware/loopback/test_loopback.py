from BrainSlice import session, Mode
import loopback
import numpy
from pathlib import Path

examples = Path(__file__).resolve().parents[1]
outdir = "x64/Debug"
firmware = examples / "firmware" / outdir / "loopback"
emulator = examples / "emulator" / outdir

# Mode.Emulator - run in emulator
# Mode.Fpga - run on Catapult FPGA
mode = Mode.Emulator

# Use x86 dll build of firmware rather then Nios binaries
x86 = False

with session(mode) as sess:
    if (x86 and mode == Mode.Emulator):
       sess.load_firmware(
           schema = str(firmware / "schema.bin"),
           dll    = str(emulator / "loopback_firmware"))
    else:
       sess.load_firmware(
           schema = str(firmware / "schema.bin"),
           inst   = str(firmware / "instructions.bin"),
           data   = str(firmware / "data.bin"))

    args = loopback.Param(
            scalar = True,
            dim = sess.sku.NATIVE_DIM)

    result = loopback.Result(scalar = False)

    v_in = numpy.random.rand(args.dim)
    o = sess.run(loopback.Loopback(args, result, Input = v_in))
    v_out = o["Output"]

    assert numpy.allclose(v_in, v_out, rtol=0.1)
    assert result.scalar == args.scalar

    print("Finished running loopback test with {firmware} firmware using {mode}"
           .format(firmware = "x86" if x86 else "Nios", mode = mode))
