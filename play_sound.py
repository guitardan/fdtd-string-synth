import numpy as np
import sounddevice as sd
import string_synth as ss

duration = 10
f0 = 432
inharmonicity = 1 # 1e-5

Fs = sd.query_devices('output')['default_samplerate']
lss = ss.LossyStiffString(
    f0,
    Fs,
    [500, 5_000],
    [duration, 0.6*duration],
    ss.ExcitationParameters(0.96, 1e-4, 0.11),
    True,
    inharmonicity
)
li = ss.LinearInterpolant(0.99, lss.N)

M = int(Fs*duration)
u = np.zeros((lss.N+1))
out = np.zeros((M,1))
for m in range(M):
    u[2:-1] = np.matmul(lss.M1,lss.u1[2:-1]) + np.matmul(lss.M2,lss.u2[2:-1])
    out[m] = ss.interpolated_read(u, li)
    lss.u2, lss.u1 = lss.u1.copy(), u.copy()

sd.play(out/abs(out).max(),Fs)