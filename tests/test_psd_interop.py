"""Independent PSD interoperability, rather than a writer/reader cancelling bugs."""
import io
import numpy as np
import pytest
from PIL import Image
from moka.formats import read_psd, read_psd_basic, write_psd


def test_independent_psd_tools_order_and_composite():
    PSDImage = pytest.importorskip("psd_tools").PSDImage
    size = (20, 20)
    back = Image.new("RGBA", size, (20, 80, 150, 255))
    front = Image.new("RGBA", (8, 8), (220, 100, 30, 255))
    layers = [("Back", back, (0, 0)), ("Front", front, (4, 5))]
    expected = back.copy(); expected.paste(front, (4, 5))
    exported = PSDImage.open(io.BytesIO(write_psd(layers, size)))
    assert [layer.name for layer in exported] == ["Back", "Front"]
    assert np.array_equal(np.asarray(exported.composite(force=True).convert("RGBA")), np.asarray(expected))
    # Construct a file using an independent writer and read it with BOTH paths.
    independent = PSDImage.new("RGBA", size)
    independent.create_pixel_layer(back, name="Back")
    independent.create_pixel_layer(front, name="Front", left=4, top=5)
    buffer = io.BytesIO(); independent.save(buffer)
    for reader in (read_psd_basic, read_psd):
        decoded, dimensions, warnings = reader(buffer.getvalue())
        assert dimensions == size
        assert [layer[0] for layer in decoded] == ["Back", "Front"]
        canvas = Image.new("RGBA", size)
        for _, image, position in decoded:
            canvas.alpha_composite(image, dest=position)
        assert np.array_equal(np.asarray(canvas), np.asarray(expected))
