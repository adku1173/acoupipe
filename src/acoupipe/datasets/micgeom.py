from acoular import MicGeom, __file__
from pathlib import Path

mics = MicGeom(file=Path(__file__).parent / 'xml' / 'tub_vogel64.xml')
tub_vogel64 = mics.pos.copy()
tub_vogel64_ap1 = tub_vogel64 / mics.aperture
