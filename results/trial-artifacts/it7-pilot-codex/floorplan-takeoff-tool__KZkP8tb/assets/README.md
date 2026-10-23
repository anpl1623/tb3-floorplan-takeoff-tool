`finish_tags.npz` contains the eight standard finish-code labels, including their frames, sampled from the supplied scan. These are used for small-vocabulary annotation recognition.

`scale_digits.npz` contains numeric glyphs sampled from the supplied graphic scale, plus rasterized variants of PyMuPDF's built-in Helvetica, bold Helvetica, Courier, and Times fonts. The font variants are already rasterized; runtime font files are unnecessary.

Neither asset contains store identifiers, plan layouts, or quantity answers. Fixture and door templates are constructed parametrically in Python.
