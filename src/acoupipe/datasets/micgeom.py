from pathlib import Path

import acoular as ac

mics = ac.MicGeom(file=Path(ac.__file__).parent / 'xml' / 'tub_vogel64.xml')
tub_vogel64 = mics.pos.copy()
tub_vogel64_ap1 = tub_vogel64 / mics.aperture
