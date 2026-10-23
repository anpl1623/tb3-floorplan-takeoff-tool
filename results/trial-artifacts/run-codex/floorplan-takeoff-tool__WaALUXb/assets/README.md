# CAD references

- `finish_features.npz`: 48 reference texture vectors and their eight finish
  labels, plus feature normalization scales. Each 34-component vector is a
  median over small, scale-normalized image patches from a finish region in
  the three supplied scans. Features describe oriented spatial frequencies
  and ink energy. They contain no dimensions, store identifiers, room
  positions or takeoff quantities.
- `cooler_front.png`: a 50-by-25-pixel reference crop of one repeated cooler
  door front, normalized to 20 pixels per foot, from the supplied store-0486
  scan. Matching rotates this crop to recognize all case directions.

Other fixture templates are drawn procedurally from the standard CAD block
geometry in `fixtures.py`. All assets are local and are read without updates
or downloads.
